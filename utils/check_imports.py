"""Cross-import verification: each module can import its dependencies."""
import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

modules = [
    # Brain tier
    ("brain.doubt_engine",      "DoubtEngine, SafetyGate, OutputGate, DoubtVerdict, load_doubt_engine"),
    ("brain.tokenizer",         "TarsTokenizer"),
    ("brain.speculative",       "SpeculativeDecoder"),
    ("brain.rrn",               None),
    # brain.mamba2
    ("brain.mamba2.model",      "TarsMamba2LM"),
    ("brain.mamba2.brain_core", "BrainCore, WaveConsolidation, GlobalWorkspace"),
    ("brain.mamba2.inference_engine", "InferenceEngine"),
    ("brain.mamba2.verification_suite", "VerificationSuite"),
    ("brain.mamba2.ssd",        "wkv_scan"),
    ("brain.mamba2.critic",     "CriticHead"),
    ("brain.mamba2.integral_auditor", "IntegralAuditor, MetaAuditor, MetaCortex, TemporalEmbedding"),
    ("brain.mamba2.thinking_chain", None),
    ("brain.mamba2.logger",     None),
    ("brain.mamba2.mole_router", None),
    ("brain.mamba2.matrix_pool", None),
    ("brain.mamba2.neuromodulator", None),
    ("brain.mamba2.novelty",    None),
    ("brain.mamba2.optimizations", None),
    ("brain.mamba2.query_router", None),
    ("brain.mamba2.self_learn", None),
    ("brain.mamba2.bitnet",     None),
    ("brain.mamba2.tars_block", None),
    ("brain.mamba2.mixture_of_depths", None),
    ("brain.mamba2.personality_adapter", None),
    ("brain.mamba2.generate_mamba", None),
    # Agent tier
    ("agent.tars_agent",        "TarsAgent"),
    ("agent.executor",          "TarsExecutor"),
    ("agent.self_learn",        None),
    ("agent.knowledge_graph",   None),
    ("agent.moira",             None),
    ("agent.document_sense",    None),
    ("agent.file_helper",       None),
    ("agent.gie",               None),
    ("agent.learning_helper",   None),
    ("agent.skill_learn",       None),
    # Memory tier
    ("memory.leann",            "LeannIndex"),
    ("memory.titans",           "TitansMemory"),
    ("memory.store",            None),
    ("memory.memo",             None),
    ("memory.search_utils",     None),
    ("memory.restore",          None),
    # Sensory
    ("sensory.voice",           None),
    ("sensory.vision",          None),
    ("sensory.ssm_vad",         None),
    ("sensory.intonation_sensor", None),
    # Tools
    ("tools",                   None),
    ("tools.web_search",        None),
    ("tools.document_tools",    None),
    ("tools.micro_agents",      None),
    ("tools.sub_agents",        None),
    ("tools.telegram_bot",      None),
    # Training
    ("training.train_doubt",    None),
    ("training.eval_doubt",     None),
    ("training.curriculum",     None),
    ("training.train_utils",    None),
    # UI
    ("ui.ws_server",            None),
    # Utils
    ("utils.logging_config",    None),
]

ok = 0
fail = 0
for mod_name, classes in modules:
    try:
        mod = __import__(mod_name, fromlist=['_'])
        if classes:
            for cls in classes.split(', '):
                if not hasattr(mod, cls.strip()):
                    print(f"MISS  {mod_name}.{cls.strip()}")
                    fail += 1
                    continue
        ok += 1
    except Exception as e:
        err_msg = str(e).split('\n')[0][:80]
        print(f"FAIL  {mod_name}: {err_msg}")
        fail += 1

print(f"\n{'='*50}")
print(f"Import check: {ok} OK, {fail} FAIL out of {ok+fail}")
if fail == 0:
    print("ALL IMPORTS CLEAN")
