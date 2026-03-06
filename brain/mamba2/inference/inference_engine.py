"""
═══════════════════════════════════════════════════════════════
  InferenceEngine — Inference Pipeline (think + generate + verify)
═══════════════════════════════════════════════════════════════

Извлечён из TarsMamba2LM (model.py) при декомпозиции T07.
Содержит:
  - think() — адаптивный inference с Integral Auditor + IDME
  - generate_speculative() — speculative decoding
  - self_verify() — self-verification
  
T08: Pipeline Error Recovery
  - Try/except вокруг каждой волны
  - NaN → nan_to_num + warning
  - OOM → skip MoLE/RAG, core-only path
  - >50% failed → partial result + error stats
  - Per-wave health check: clamp при extreme norms
"""

import copy
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any

from brain.mamba2.critic import TaskSpec
from brain.mamba2.query_router import ProgressiveMemoryManager

logger = logging.getLogger("Tars.InferenceEngine")


class InferenceEngine:
    """
    InferenceEngine — адаптивный inference ТАРС.
    
    Не является nn.Module (не владеет параметрами).
    Использует BrainCore и VerificationSuite через ссылки.
    
    Методы:
      - think() — полный inference pipeline с dynamic depth
      - generate_speculative() — speculative decoding
      - self_verify() — проверка consistency ответа
    """
    
    def __init__(self, core, verification):
        """
        Args:
            core: BrainCore instance (owns all trainable params)
            verification: VerificationSuite instance (auditors + critics)
        """
        self.core = core
        self.verification = verification
        self.logger = logging.getLogger("Tars.InferenceEngine")
    
    def think(
        self,
        input_ids: torch.Tensor,
        query_text: str = "",
        memory_vec: Optional[torch.Tensor] = None,
        rag_state: Optional[torch.Tensor] = None,
        force_deep: bool = False,
        max_expansion_rounds: int = 12,
        reflex_ctx: Any = None,
        supplement_queue=None,
        task_spec: Any = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Адаптивный forward с Integral Auditor и Speculative Matrix Routing.
        
        T08: Error Recovery
          - Try/except around each wave
          - NaN → nan_to_num
          - Per-wave health check (norm clamping)
          - >50% waves failed → partial result
        """
        start_time = time.time()
        core = self.core
        verif = self.verification
        
        # ═══ PHASE 0: Init inference modules ═══
        verif.ensure_inference_modules()
        
        # Reset auditors
        verif.reset()
        core.matrix_pool.reset()
        core.thinking_chain.reset()
        
        # ═══ Cerebellum: set task spec ═══
        if task_spec is not None:
            verif.set_task_spec(task_spec)
        else:
            verif.set_task_spec(TaskSpec(query=query_text))
        
        # Meta-Auditor: task type and threshold
        task_type, p_threshold = verif.classify_task(query_text)
        
        # MetaCortex adaptation
        try:
            with torch.no_grad():
                if not hasattr(core, '_tokenizer'):
                    from brain.tokenizer import TarsTokenizer
                    core._tokenizer = TarsTokenizer(mode="auto")
                query_token_ids = core._tokenizer.encode(query_text)[:256]
                query_ids = torch.tensor([query_token_ids], dtype=torch.long).to(input_ids.device)
                query_emb = core.embedding(query_ids).mean(dim=1)
                error_pred = verif.predict_error(query_emb)
                p_threshold = verif.adapt_threshold(p_threshold, task_type, error_pred)
        except Exception:
            error_pred = 0.5
        
        verif.integral_auditor.threshold = p_threshold
        core.thinking_logger.start_session(query_text, task_type, p_threshold)
        
        # ═══ Adaptive Depth from ReflexContext ═══
        estimated_depth = core.n_layers
        reflex_urgency = 0.0
        
        if reflex_ctx is not None:
            estimated_depth = min(
                getattr(reflex_ctx, 'estimated_depth', core.n_layers),
                core.n_layers
            )
            if memory_vec is None and getattr(reflex_ctx, 'memory_vec', None) is not None:
                memory_vec = reflex_ctx.memory_vec
            if getattr(reflex_ctx, 'needs_idme', False):
                max_expansion_rounds = getattr(reflex_ctx, 'max_expansion_rounds', 12)
            else:
                max_expansion_rounds = 2
            reflex_urgency = getattr(reflex_ctx, 'urgency', 0.0)
        
        TASK_MAX_ROUNDS = {
            "chat": 2, "action": 2, "code": 6,
            "math": 8, "deep": 20, "infinite": 100,
        }
        task_max = TASK_MAX_ROUNDS.get(task_type, 4)
        max_expansion_rounds = min(max_expansion_rounds, task_max)
        if force_deep:
            max_expansion_rounds = 100
            estimated_depth = core.n_layers
        
        # ═══ 1. Parallel Wave Depth ═══
        x = core.embedding(input_ids)
        
        # QueryRouter
        query_emb = x.mean(dim=1).detach()
        router_decision = core.query_router(query_emb)
        
        # Progressive Memory Manager
        pmm = ProgressiveMemoryManager()
        pmm.set_initial_memory(memory_vec=memory_vec, rag_state=rag_state)
        
        rag_skipped = False
        if not router_decision["needs_rag"].any():
            rag_skipped = True
        else:
            source = core.query_router.get_primary_source(router_decision["source_weights"])
            urgency_val = router_decision["urgency"].mean().item()
            source_names = ["leann", "titans", "web", "history"]
            weights = router_decision["source_weights"][0].tolist()
            requested = [n for n, w in zip(source_names, weights) if w > 0.1]
            pmm.register_request(requested, urgency=urgency_val)
        
        wkv_states = [None] * core.n_layers
        x_prevs = [None] * core.n_layers
        ssd_states = [None] * core.n_layers
        conv_states = [None] * core.n_layers
        h_prev = x.mean(dim=1).detach()
        
        block_stats = []
        blocks_executed = 0
        surprise_signals = []
        converged_early = False
        wave_count = 0
        per_wave_experts = []
        supplement_injected = False
        supplement_extra_waves = 0
        rag_injection_wave = -1
        failed_waves = 0  # T08: error recovery counter
        
        max_waves = core.n_layers // 2
        
        # ThoughtCache shortcut
        cache_skip_waves = 0
        if hasattr(core, 'thinking_chain'):
            try:
                cache_skip_waves = core.thinking_chain.try_cache_shortcut(x.mean(dim=1))
                cached_mem = core.thinking_chain.get_cached_memory()
                if cached_mem is not None and cache_skip_waves > 0:
                    memory_vec = cached_mem.to(x.device)
            except Exception:
                cache_skip_waves = 0
        
        wave_outputs = {}
        critic_result = None
        ia_result = {"p": 0.0, "r_squared": 0.0, "converged": False, "f_t": 0.0}
        
        for wave_idx in range(max_waves):
            if blocks_executed >= estimated_depth and not force_deep:
                break
            
            if wave_idx < cache_skip_waves:
                wave_count += 1
                blocks_executed += 2
                continue
            
            wave_count += 1
            b_left = wave_idx * 2
            b_right = wave_idx * 2 + 1
            
            if b_right >= core.n_layers:
                break
            
            # ═══ T08: Try/except around each wave ═══
            try:
                # Progressive Memory Injection
                if pmm.check_and_inject(wave_idx):
                    rag_injection_wave = wave_idx
                    memory_vec = pmm.get_memory_vec()
                    rag_state = pmm.get_rag_state()
                    verif.integral_auditor.reset()
                    h_prev = x.mean(dim=1).detach()
                else:
                    memory_vec = pmm.get_memory_vec()
                    rag_state = pmm.get_rag_state()
                
                # Two blocks in parallel
                x_left, wkv_states[b_left], x_prevs[b_left], stats_l, \
                    ssd_states[b_left], conv_states[b_left] = core.blocks[b_left](
                    x, wkv_states[b_left], x_prevs[b_left], memory_vec, rag_state,
                    ssd_states[b_left], conv_states[b_left]
                )
                x_right, wkv_states[b_right], x_prevs[b_right], stats_r, \
                    ssd_states[b_right], conv_states[b_right] = core.blocks[b_right](
                    x, wkv_states[b_right], x_prevs[b_right], memory_vec, rag_state,
                    ssd_states[b_right], conv_states[b_right]
                )
                block_stats.extend([stats_l, stats_r])
                blocks_executed += 2
                
                wave_experts = {
                    "wave": wave_count,
                    "left": stats_l.get("mole_experts", []),
                    "right": stats_r.get("mole_experts", []),
                }
                per_wave_experts.append(wave_experts)
                
                # ═══ WaveConsolidation ═══
                x, merge_alpha = core.wave_consolidations[wave_idx](
                    x_left, x_right, stats_l, stats_r
                )
                
                # Cross-Wave Residual
                wave_outputs[wave_idx] = x.detach()
                skip_from = wave_idx - 3
                if skip_from in wave_outputs:
                    x = x + 0.1 * wave_outputs[skip_from]
                
                wave_experts["merge_alpha"] = merge_alpha
                
                # ═══ T08: NaN/Inf Guard ═══
                if torch.isnan(x).any() or torch.isinf(x).any():
                    self.logger.warning(
                        f"NaN/Inf at wave {wave_count}! Reverting to previous state."
                    )
                    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
                    failed_waves += 1
                
                # ═══ T08: Per-wave health check ═══
                x_norm = x.float().norm().item()
                if x_norm < 0.01 or x_norm > 100:
                    x = x.clamp(-10, 10)
                    self.logger.warning(f"Wave {wave_count}: norm={x_norm:.4f}, clamped")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # ═══ T08: OOM Recovery — skip heavy modules ═══
                    self.logger.error(f"OOM at wave {wave_count}, using core-only path")
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    try:
                        x_left, wkv_states[b_left], x_prevs[b_left], stats_l, \
                            ssd_states[b_left], conv_states[b_left] = core.blocks[b_left](
                            x, wkv_states[b_left], x_prevs[b_left], None, None,
                            ssd_states[b_left], conv_states[b_left]
                        )
                        x_right, wkv_states[b_right], x_prevs[b_right], stats_r, \
                            ssd_states[b_right], conv_states[b_right] = core.blocks[b_right](
                            x, wkv_states[b_right], x_prevs[b_right], None, None,
                            ssd_states[b_right], conv_states[b_right]
                        )
                        x, _ = core.wave_consolidations[wave_idx](x_left, x_right)
                        blocks_executed += 2
                    except Exception:
                        failed_waves += 1
                        self.logger.error(f"Core-only fallback also failed at wave {wave_count}")
                        continue
                else:
                    failed_waves += 1
                    self.logger.error(f"Wave {wave_count} failed: {e}")
                    continue
            except Exception as e:
                failed_waves += 1
                self.logger.error(f"Wave {wave_count} failed: {e}")
                continue
            
            # ═══ T08: >50% waves failed → break early ═══
            if failed_waves > max_waves * 0.5:
                self.logger.error(
                    f">{50}% waves failed ({failed_waves}/{wave_count}), returning partial result"
                )
                break
            
            # Surprise signals
            for stats in [stats_l, stats_r]:
                if stats.get("surprise", 0.0) > 0.3:
                    surprise_signals.append({
                        "layer": stats["layer_idx"],
                        "surprise": stats["surprise"],
                        "mem_relevance": stats.get("mem_relevance", 0.0),
                    })
            
            # Integral Auditor: convergence check
            h_curr = x.mean(dim=1).detach()
            ia_result = verif.observe(h_curr, h_prev)
            h_prev = h_curr
            
            core.thinking_logger.log_step(blocks_executed - 1, {
                "p": ia_result["p"], "r_squared": ia_result["r_squared"],
                "f_t": ia_result["f_t"], "converged": ia_result["converged"],
                "wave": wave_count, "depth": blocks_executed, "width": 2,
            })
            
            # Convergence check
            ia_converged = ia_result["converged"] and wave_count >= 3
            tc_skip = hasattr(core, 'thinking_chain') and core.thinking_chain.should_skip_remaining()
            
            strict_tasks = {"code", "math", "deep"}
            if task_type in strict_tasks:
                ia_quality = ia_converged and ia_result["r_squared"] > 0.92
                tc_quality = tc_skip
                critic_ok = critic_result is not None and critic_result.get("score", 0) > 0.6
                votes = sum([ia_quality, tc_quality, critic_ok])
                should_stop = ia_converged and votes >= 2
            else:
                should_stop = ia_converged or tc_skip
            
            if should_stop:
                converged_early = True
                break
            
            # CriticHead
            critic_result = None
            if wave_idx < max_waves - 1:
                try:
                    h_query_mean = core.embedding(input_ids).mean(dim=1).detach()
                    critic_result = verif.evaluate_wave(
                        h_curr, h_query_mean, wave_idx, memory_vec
                    )
                    
                    if critic_result["feedback_vec"] is not None and memory_vec is not None:
                        fb = critic_result["feedback_vec"]
                        if fb.shape[-1] != memory_vec.shape[-1]:
                            if hasattr(core, 'to_memory_space'):
                                fb = core.to_memory_space(fb)
                        if fb.shape == memory_vec.shape:
                            memory_vec = memory_vec + 0.3 * fb
                    
                    ia_strong = ia_result["p"] > 0.8 and ia_result["r_squared"] > 0.7
                    if critic_result["should_stop"] and ia_strong and wave_count >= 3:
                        converged_early = True
                        break
                except Exception as e:
                    self.logger.debug(f"Critic eval error: {e}")
            
            # Spine update
            spine_updated = False
            if wave_idx < max_waves - 1:
                if hasattr(core, 'to_memory_space'):
                    try:
                        pmm.update_spine(h_curr, core.to_memory_space, alpha=0.3)
                        memory_vec = pmm.get_memory_vec()
                        spine_updated = True
                    except Exception:
                        pass
                
                # Titans surprise feedback
                if hasattr(core, 'titans_memory') and core.titans_memory is not None:
                    wave_surprises = [s for s in surprise_signals
                                     if s["layer"] >= blocks_executed - 2]
                    if wave_surprises:
                        try:
                            h_for_titans = core.to_memory_space(h_curr)
                            core.titans_memory.forward(h_for_titans)
                        except Exception:
                            pass
                
                # ThinkingChain
                if memory_vec is not None and hasattr(core, 'thinking_chain'):
                    try:
                        memory_vec, thought_info = core.thinking_chain.step(
                            h_curr, memory_vec, wave_idx, max_waves
                        )
                    except Exception:
                        pass
            
            # Supplement injection
            if supplement_queue is not None:
                try:
                    supplement = supplement_queue.get_nowait()
                    sup_text = supplement.get("text", "")
                    sup_tokens = supplement.get("tokens")
                    
                    if sup_text or sup_tokens is not None:
                        if sup_tokens is None:
                            sup_ids = input_ids
                        else:
                            sup_ids = sup_tokens.to(x.device)
                        
                        x_sup = core.embedding(sup_ids)
                        xs_l, _, _, _, _, _ = core.blocks[b_left](
                            x_sup, None, None, memory_vec, rag_state, None, None
                        )
                        xs_r, _, _, _, _, _ = core.blocks[b_right](
                            x_sup, None, None, memory_vec, rag_state, None, None
                        )
                        x_sup_merged, _ = core.wave_consolidations[wave_idx](
                            xs_l, xs_r, {}, {}
                        )
                        
                        h_sup = x_sup_merged.mean(dim=1).detach()
                        if hasattr(core, 'to_memory_space'):
                            pmm.update_spine(h_sup, core.to_memory_space, alpha=0.5)
                            memory_vec = pmm.get_memory_vec()
                        
                        verif.integral_auditor.reset()
                        converged_early = False
                        h_prev = h_sup
                        supplement_injected = True
                        supplement_extra_waves = 3
                except Exception as e:
                    if "Empty" not in str(type(e).__name__):
                        self.logger.warning(f"Supplement injection error: {e}")
        
        # Extra waves after supplement
        if supplement_injected and supplement_extra_waves > 0:
            for extra_idx in range(supplement_extra_waves):
                b_l = extra_idx * 2
                b_r = extra_idx * 2 + 1
                if b_r >= core.n_layers:
                    break
                
                wave_count += 1
                try:
                    x_left, wkv_states[b_l], x_prevs[b_l], stats_l, \
                        ssd_states[b_l], conv_states[b_l] = core.blocks[b_l](
                        x, wkv_states[b_l], x_prevs[b_l], memory_vec, rag_state,
                        ssd_states[b_l], conv_states[b_l]
                    )
                    x_right, wkv_states[b_r], x_prevs[b_r], stats_r, \
                        ssd_states[b_r], conv_states[b_r] = core.blocks[b_r](
                        x, wkv_states[b_r], x_prevs[b_r], memory_vec, rag_state,
                        ssd_states[b_r], conv_states[b_r]
                    )
                    blocks_executed += 2
                    
                    x, _ = core.wave_consolidations[extra_idx](
                        x_left, x_right, stats_l, stats_r
                    )
                    
                    h_curr = x.mean(dim=1).detach()
                    if hasattr(core, 'to_memory_space'):
                        try:
                            h_for_mem = core.to_memory_space(h_curr)
                            memory_vec = 0.7 * memory_vec + 0.3 * h_for_mem.detach()
                        except Exception:
                            pass
                    
                    ia_result = verif.observe(h_curr, h_prev)
                    h_prev = h_curr
                    
                    if ia_result["converged"] and ia_result["r_squared"] > 0.88:
                        converged_early = True
                        break
                except Exception as e:
                    self.logger.warning(f"Extra wave {extra_idx} failed: {e}")
                    failed_waves += 1
        
        # Titans feedback
        if hasattr(core, 'titans_memory') and core.titans_memory is not None:
            if surprise_signals:
                try:
                    h_for_titans = core.to_memory_space(x.mean(dim=1).detach())
                    core.titans_memory.forward(h_for_titans)
                except Exception:
                    pass
        
        # All experts
        all_expert_names = []
        for we in per_wave_experts:
            all_expert_names.extend(we.get("left", []))
            all_expert_names.extend(we.get("right", []))
        
        # ═══ 2. IDME Speculative Matrix Routing ═══
        h_curr = x.mean(dim=1).detach()
        ia_result = verif.observe(h_curr, h_prev)
        
        expansion_round = 0
        total_matrices_recruited = 0
        branches_tested = 0
        branches_won = []
        
        prev_p = ia_result.get("p", 0.0)
        no_improve_count = 0
        N_CANDIDATES = 3
        
        while not ia_result["converged"] and expansion_round < max_expansion_rounds:
            expansion_round += 1
            h_prev = h_curr
            
            candidates, indices = core.matrix_pool.select(h_curr.mean(0), k=N_CANDIDATES)
            total_matrices_recruited += len(candidates)
            branches_tested += len(candidates)
            
            if not candidates:
                available = core.matrix_pool.total_available
                used = len(getattr(core.matrix_pool, 'used_mask', []))
                if available <= used + N_CANDIDATES:
                    try:
                        core.matrix_pool._lazy_expand(4, h_curr.mean(0))
                        candidates, indices = core.matrix_pool.select(h_curr.mean(0), k=N_CANDIDATES)
                        total_matrices_recruited += len(candidates)
                        branches_tested += len(candidates)
                    except Exception:
                        pass
                if not candidates:
                    break
            
            best_p = prev_p
            best_x = None
            best_idx = -1
            h_state = x.mean(dim=1)
            
            for matrix, idx in zip(candidates, indices):
                h_clone = h_state.clone()
                h_refined = matrix(h_clone)
                h_gated, _ = core.novelty_gate(h_clone, h_refined)
                
                with torch.no_grad():
                    f_t = (h_gated - h_state).float().norm().item()
                    candidate_p = prev_p + (1.0 / max(f_t, 1e-8)) * 0.01
                
                if candidate_p > best_p:
                    best_p = candidate_p
                    best_x = h_gated
                    best_idx = idx
            
            if best_x is not None and best_p > prev_p:
                if torch.isnan(best_x).any() or torch.isinf(best_x).any():
                    no_improve_count += 1
                    continue
                x = x + 0.1 * (best_x - h_state).unsqueeze(1)
                h_curr = x.mean(dim=1).detach()
                ia_result = verif.observe(h_curr, h_prev)
                
                if best_idx >= 0:
                    core.matrix_pool.recirculate(best_idx, max(0, ia_result["p"] - prev_p))
                    branches_won.append((best_idx, ia_result["p"] - prev_p))
                
                hankel_result = verif.observe_hankel(h_curr)
                if hankel_result["collapsed"] and hankel_result["collapse_count"] >= 3:
                    break
            else:
                no_improve_count += 1
                h_curr = x.mean(dim=1).detach()
                ia_result = verif.observe(h_curr, h_prev)
            
            if ia_result["p"] <= prev_p + 0.01:
                no_improve_count += 1
                if no_improve_count >= 2:
                    break
            else:
                no_improve_count = 0
            prev_p = ia_result["p"]
        
        # RAG Completion Verification
        rag_verification = {}
        try:
            delivery_report = pmm.get_delivery_report()
            rag_verification = verif.verify_rag_completion(
                delivery_report=delivery_report,
                memory_vec=memory_vec,
                pmm=pmm,
                matrix_pool=core.matrix_pool,
                x=x,
                h_prev=h_prev,
                integral_auditor=verif.integral_auditor,
                novelty_gate=core.novelty_gate,
                from_memory_space=core.from_memory_space,
                to_memory_space=core.to_memory_space,
            )
            if rag_verification.get("updated_memory_vec") is not None:
                memory_vec = rag_verification["updated_memory_vec"]
            if rag_verification.get("updated_x") is not None:
                x = rag_verification["updated_x"]
            total_matrices_recruited += rag_verification.get("matrices_recruited", 0)
        except Exception as e:
            self.logger.debug(f"Cerebellum RAG verification error: {e}")
        
        # ═══ 3. Output ═══
        if memory_vec is not None and hasattr(core, 'from_memory_space'):
            try:
                mem_signal = core.from_memory_space(memory_vec)
                x = x + 0.1 * mem_signal.unsqueeze(1)
            except Exception:
                pass
        
        x = core.norm_f(x)
        logits = core.lm_head(x)
        
        # Confidence-Gated Output
        if hasattr(core, 'thinking_chain'):
            try:
                logits = core.thinking_chain.apply_confidence_gate(logits)
                if memory_vec is not None:
                    core.thinking_chain.update_session_memory(memory_vec)
            except Exception:
                pass
        
        # Output Sanity Check
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            self.logger.warning("CRITICAL: NaN/Inf in output logits! Replacing with uniform.")
            logits = torch.zeros_like(logits)
        else:
            logit_std = logits.float().std().item()
            if logit_std < 1e-6:
                logits = logits + torch.randn_like(logits) * 0.01
        
        total_time = (time.time() - start_time) * 1000
        
        # ThinkingChain finalize
        tc_summary = {}
        if hasattr(core, 'thinking_chain'):
            try:
                avg_surprise = 0.0
                if surprise_signals:
                    avg_surprise = sum(s["surprise"] for s in surprise_signals) / len(surprise_signals)
                core.thinking_chain.finalize(
                    query_embedding=x.mean(dim=1).detach(),
                    final_memory=memory_vec,
                    surprise_level=avg_surprise,
                    task_type=task_type,
                )
            except Exception:
                pass
            tc_summary = core.thinking_chain.get_summary()
        
        stats = {
            "task_type": task_type,
            "p_threshold": p_threshold,
            "final_p": ia_result["p"],
            "r_squared": ia_result.get("r_squared", 0),
            "converged": ia_result["converged"],
            "converged_early": converged_early,
            "total_blocks": core.n_layers,
            "blocks_executed": blocks_executed,
            "waves": wave_count,
            "estimated_depth": estimated_depth,
            "expansion_rounds": expansion_round,
            "matrices_recruited": total_matrices_recruited,
            "total_matrices": core.n_layers + total_matrices_recruited,
            "branches_tested": branches_tested,
            "branches_won": len(branches_won),
            "per_wave_experts": per_wave_experts,
            "hankel_collapses": verif.hankel.collapse_count,
            "surprise_layers": len(surprise_signals),
            "supplement_injected": supplement_injected,
            "supplement_extra_waves": supplement_extra_waves if supplement_injected else 0,
            "rwkv_state_size_mb": sum(s.numel() * 4 / 1024 / 1024 for s in wkv_states if s is not None),
            "total_ms": total_time,
            # T08: error recovery stats
            "failed_waves": failed_waves,
            "partial_result": failed_waves > max_waves * 0.5,
            # ThinkingChain
            "thinking_chain": tc_summary,
            "tc_confidence": tc_summary.get("final_confidence", 0),
            "tc_phases": tc_summary.get("phases", []),
            "tc_retrieval_count": tc_summary.get("retrieval_count", 0),
            # QueryRouter & PMM
            "rag_skipped": rag_skipped,
            "rag_injection_wave": rag_injection_wave,
            "query_router_needs_rag": not rag_skipped,
            "query_router_urgency": router_decision["urgency"].mean().item(),
            "query_router_source": (
                core.query_router.get_primary_source(router_decision["source_weights"])
                if not rag_skipped else "none"
            ),
            # RAG Verification
            "rag_verification": rag_verification,
            "rag_all_loaded": rag_verification.get("rag_complete", True),
            "rag_continuation_rounds": rag_verification.get("rounds_used", 0),
            "rag_missing_sources": rag_verification.get("missing_sources", []),
        }
        
        core.thinking_logger.end_session(total_time, "")
        
        self.logger.info(
            f"Think: {task_type} | p={stats['final_p']:.3f} | "
            f"conf={stats['tc_confidence']:.2f} | "
            f"failed_waves={failed_waves}"
        )
        
        return logits, stats
    
    @torch.no_grad()
    def generate_speculative(
        self,
        prompt_ids: torch.Tensor,
        max_tokens: int = 128,
        n_draft: int = 4,
        temperature: float = 0.7,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Speculative Decoding with draft → verify → accept/reject."""
        core = self.core
        core.eval()
        device = prompt_ids.device
        
        core.reset_cache()
        logits = core.step(prompt_ids)
        
        generated = prompt_ids.clone()
        n_accepted_total = 0
        n_drafted_total = 0
        
        for _ in range(max_tokens):
            if generated.shape[1] - prompt_ids.shape[1] >= max_tokens:
                break
            
            cache_snapshot = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in core._gen_cache.items()
            } if isinstance(core._gen_cache, dict) else copy.deepcopy(core._gen_cache)
            
            last_logits = logits[:, -1, :]
            first_token = core._sample(last_logits, temperature, top_k)
            
            draft_tokens = [first_token]
            draft_logits_list = [last_logits]
            draft_input = first_token
            
            for _ in range(n_draft - 1):
                step_logits = core.step(draft_input.unsqueeze(1))
                draft_next = core._sample(step_logits[:, -1, :], temperature, top_k)
                draft_tokens.append(draft_next)
                draft_logits_list.append(step_logits[:, -1, :])
                draft_input = draft_next
            
            core._gen_cache = cache_snapshot
            
            verify_input = torch.stack(draft_tokens).unsqueeze(0)
            verify_logits = core.step(verify_input)
            
            n_accepted = 1
            for i in range(len(draft_tokens) - 1):
                verify_token = verify_logits[:, i, :].argmax(dim=-1)
                if verify_token.item() == draft_tokens[i + 1].item():
                    n_accepted += 1
                else:
                    draft_tokens[i + 1] = verify_token
                    n_accepted += 1
                    break
            
            n_drafted_total += len(draft_tokens)
            n_accepted_total += n_accepted
            
            accepted = torch.stack(draft_tokens[:n_accepted]).unsqueeze(0)
            generated = torch.cat([generated, accepted], dim=1)
            
            if n_accepted < len(draft_tokens):
                core._gen_cache = cache_snapshot
                logits = core.step(accepted)
            else:
                logits = verify_logits[:, n_accepted - 1:n_accepted, :]
        
        return generated
    
    @torch.no_grad()
    def self_verify(self, input_ids, generated_ids, memory_vec=None):
        """Self-Verification: re-run answer through 2 waves."""
        core = self.core
        core.eval()
        
        combined = torch.cat([input_ids, generated_ids], dim=1)
        x = core.embedding(combined)
        
        for wave_idx in range(min(2, core.n_layers // 2)):
            b_left = wave_idx * 2
            b_right = wave_idx * 2 + 1
            if b_right >= core.n_layers:
                break
            
            x_left, _, _, _, _, _ = core.blocks[b_left](
                x, None, None, memory_vec, None, None, None
            )
            x_right, _, _, _, _, _ = core.blocks[b_right](
                x, None, None, memory_vec, None, None, None
            )
            x, _ = core.wave_consolidations[wave_idx](x_left, x_right)
        
        L_q = input_ids.shape[1]
        answer_hidden = x[:, L_q:, :].mean(dim=1)
        query_hidden = x[:, :L_q, :].mean(dim=1)
        
        consistency = F.cosine_similarity(
            answer_hidden, query_hidden, dim=-1
        ).item()
        consistency = (consistency + 1.0) / 2.0
        should_regenerate = consistency < 0.8
        
        return consistency, should_regenerate
