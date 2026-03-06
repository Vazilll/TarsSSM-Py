/* ═══════════════════════════════════════════════════
   ТАРС — Voice-First Engine
   Web Speech API + Waveform + Auto-Listen
   ═══════════════════════════════════════════════════ */

// ═══ State ═══
let state = 'idle'; // idle | listening | thinking | speaking
let recognition = null;
let synthesis = window.speechSynthesis;
let audioCtx = null;
let analyser = null;
let isProcessing = false;

// ═══ DOM ═══
const core = document.getElementById('core');
const coreIcon = document.getElementById('coreIcon');
const responseText = document.getElementById('responseText');
const transcript = document.getElementById('transcript');
const textInput = document.getElementById('textInput');
const waveformCanvas = document.getElementById('waveform');
const waveformCtx = waveformCanvas.getContext('2d');
const statusText = document.getElementById('statusText');
const statusTime = document.getElementById('statusTime');
const hints = document.getElementById('hints');
const thinkingSteps = document.getElementById('thinkingSteps');

// ═══ Init ═══
document.addEventListener('DOMContentLoaded', () => {
    updateClock();
    setInterval(updateClock, 10000);
    initSpeechRecognition();
    setupInteractions();

    // Auto-greet after 1 second
    setTimeout(() => {
        execCmd('мой день');
    }, 1000);
});

// ═══ Clock ═══
function updateClock() {
    const now = new Date();
    statusTime.textContent = now.toLocaleTimeString('ru-RU', {
        hour: '2-digit', minute: '2-digit'
    });
}

// ═══ Speech Recognition (Web Speech API) ═══
function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
        statusText.textContent = 'VOICE NOT SUPPORTED';
        return;
    }

    recognition = new SpeechRecognition();
    recognition.lang = 'ru-RU';
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
        setState('listening');
        transcript.textContent = '...слушаю';
    };

    recognition.onresult = (event) => {
        let interimText = '';
        let finalText = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
            const t = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
                finalText += t;
            } else {
                interimText += t;
            }
        }

        // Show interim results
        if (interimText) {
            transcript.textContent = interimText;
        }

        // Process final result
        if (finalText) {
            transcript.textContent = finalText;
            processInput(finalText.trim());
        }
    };

    recognition.onerror = (event) => {
        if (event.error === 'no-speech') {
            setState('idle');
            transcript.textContent = '';
        } else if (event.error !== 'aborted') {
            console.log('Speech error:', event.error);
            setState('idle');
        }
    };

    recognition.onend = () => {
        if (state === 'listening') {
            setState('idle');
        }
    };
}

// ═══ States ═══
function setState(newState) {
    state = newState;
    core.className = 'core';

    switch (newState) {
        case 'idle':
            coreIcon.textContent = '◉';
            statusText.textContent = 'TARS ONLINE';
            break;
        case 'listening':
            core.classList.add('listening');
            coreIcon.textContent = '●';
            statusText.textContent = 'LISTENING';
            startWaveform();
            break;
        case 'thinking':
            core.classList.add('thinking');
            coreIcon.textContent = '◎';
            statusText.textContent = 'THINKING';
            stopWaveform();
            break;
        case 'speaking':
            core.classList.add('speaking');
            coreIcon.textContent = '◉';
            statusText.textContent = 'SPEAKING';
            startWaveform();
            break;
    }
}

// ═══ Start Listening ═══
function startListening() {
    if (state === 'listening' || state === 'thinking' || isProcessing) return;

    // Stop speaking if active
    synthesis.cancel();

    try {
        recognition.start();
    } catch (e) {
        // Already started
    }
}

function stopListening() {
    try {
        recognition.stop();
    } catch (e) { }
}

// ═══ Process Input ═══
async function processInput(text) {
    if (!text || isProcessing) return;
    isProcessing = true;

    stopListening();
    setState('thinking');

    // Hide hints, clear previous
    hints.style.opacity = '0';
    responseText.textContent = '';
    clearThinking();

    // Start showing thinking steps IMMEDIATELY (client-side)
    thinkingSteps.classList.add('active');
    const clientSteps = [
        'parsing input...',
        `> "${text.slice(0, 35)}"`,
        'analyzing intent...',
        'checking subsystems...',
    ];

    let stepIdx = 0;
    const stepTimer = setInterval(() => {
        if (stepIdx >= clientSteps.length) {
            clearInterval(stepTimer);
            return;
        }
        addThinkStep(clientSteps[stepIdx]);
        stepIdx++;
    }, 300);

    // API call runs IN PARALLEL with thinking animation
    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });

        const data = await res.json();

        // Stop client steps
        clearInterval(stepTimer);

        // Show server thinking steps (real ones)
        if (data.thinking && data.thinking.length > 0) {
            for (const step of data.thinking) {
                // Skip duplicates of client steps
                if (clientSteps.includes(step)) continue;
                addThinkStep(step);
                await sleep(250);
            }
        }

        // Pause to let user see final step
        await sleep(400);

        // Fade out thinking, show response
        await fadeOutThinking();

        if (data.response) {
            showResponse(data.response);
            speak(data.response);
        }
    } catch (err) {
        clearInterval(stepTimer);
        clearThinking();
        showResponse('connection error');
        setState('idle');
    }

    isProcessing = false;
}

// ═══ Thinking Steps Display ═══
function addThinkStep(text) {
    // Dim all previous steps
    thinkingSteps.querySelectorAll('.think-step:not(.dim)').forEach(el => {
        el.classList.add('dim');
    });

    // Add new step
    const step = document.createElement('div');
    step.className = 'think-step';
    step.textContent = text;
    thinkingSteps.appendChild(step);

    // Keep only last 5 steps visible
    const all = thinkingSteps.querySelectorAll('.think-step');
    if (all.length > 5) {
        all[0].remove();
    }
}

function fadeOutThinking() {
    return new Promise((resolve) => {
        thinkingSteps.classList.add('done');
        setTimeout(() => {
            thinkingSteps.innerHTML = '';
            thinkingSteps.className = 'thinking-steps';
            resolve();
        }, 500);
    });
}

function clearThinking() {
    thinkingSteps.innerHTML = '';
    thinkingSteps.className = 'thinking-steps';
}

function sleep(ms) {
    return new Promise(r => setTimeout(r, ms));
}

// ═══ Show Response ═══
function showResponse(text) {
    // Clean response for display
    const clean = text
        .replace(/═+/g, '—')
        .replace(/\*\*(.+?)\*\*/g, '$1');

    responseText.textContent = clean;
    responseText.className = 'response-text';

    // Add 'long' class for multi-line responses
    if (clean.split('\n').length > 4 || clean.length > 200) {
        responseText.classList.add('long');
    }

    // Re-trigger animation
    responseText.style.animation = 'none';
    requestAnimationFrame(() => {
        responseText.style.animation = '';
    });
}

// ═══ Text-to-Speech ═══
function speak(text) {
    // Clean text for speech
    const cleanText = text
        .replace(/[═─│┌┐└┘├┤┬┴┼]/g, '')
        .replace(/[📊📅🍅📝🔄💰🕸💻🔍📁📋🎙🎯❓🤖✅❌⚠️💡🔔☕🏆📈📉🌅☀️🌙🟢🔴🟡█░]/g, '')
        .replace(/\[.*?\]/g, '')
        .replace(/\n{2,}/g, '. ')
        .replace(/\n/g, '. ')
        .replace(/\s{2,}/g, ' ')
        .trim();

    if (!cleanText || cleanText.length < 2) {
        setState('idle');
        return;
    }

    // Limit speech length
    const shortText = cleanText.length > 500 ? cleanText.slice(0, 500) + '...' : cleanText;

    const utterance = new SpeechSynthesisUtterance(shortText);
    utterance.lang = 'ru-RU';
    utterance.rate = 1.05;
    utterance.pitch = 0.9;

    // Try to find Russian voice
    const voices = synthesis.getVoices();
    const ruVoice = voices.find(v => v.lang.startsWith('ru'));
    if (ruVoice) utterance.voice = ruVoice;

    utterance.onstart = () => setState('speaking');
    utterance.onend = () => {
        setState('idle');
        // Auto-listen after speaking (like real TARS)
        setTimeout(() => {
            // Don't auto-listen — wait for user to tap or speak
        }, 500);
    };
    utterance.onerror = () => setState('idle');

    synthesis.speak(utterance);
}

// ═══ Waveform Visualization ═══
let waveformFrame = null;

function startWaveform() {
    if (waveformFrame) return;
    drawWaveform();
}

function stopWaveform() {
    if (waveformFrame) {
        cancelAnimationFrame(waveformFrame);
        waveformFrame = null;
    }
    // Clear canvas
    waveformCtx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
}

function drawWaveform() {
    const w = waveformCanvas.width;
    const h = waveformCanvas.height;
    const mid = h / 2;

    waveformCtx.clearRect(0, 0, w, h);
    waveformCtx.beginPath();
    waveformCtx.strokeStyle = state === 'listening'
        ? 'rgba(52, 211, 153, 0.4)'
        : 'rgba(129, 140, 248, 0.4)';
    waveformCtx.lineWidth = 1.5;

    const t = Date.now() / 1000;
    const bars = 40;
    const barW = w / bars;

    for (let i = 0; i < bars; i++) {
        const x = i * barW + barW / 2;

        // Generate organic waveform
        const amp = state === 'listening'
            ? (Math.sin(t * 3 + i * 0.3) * 0.4 + Math.sin(t * 7 + i * 0.5) * 0.3 + Math.random() * 0.3) * 20
            : (Math.sin(t * 4 + i * 0.4) * 0.5 + Math.sin(t * 9 + i * 0.3) * 0.3) * 25;

        waveformCtx.moveTo(x, mid - amp);
        waveformCtx.lineTo(x, mid + amp);
    }

    waveformCtx.stroke();
    waveformFrame = requestAnimationFrame(drawWaveform);
}

// ═══ Interactions ═══
function setupInteractions() {
    // Click core to start listening
    core.addEventListener('click', () => {
        if (state === 'speaking') {
            synthesis.cancel();
            setState('idle');
        } else if (state === 'listening') {
            stopListening();
        } else {
            startListening();
        }
    });

    // Text input — Enter to send
    textInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            const text = textInput.value.trim();
            if (text) {
                transcript.textContent = text;
                textInput.value = '';
                processInput(text);
            }
        }
    });

    // Keyboard shortcut: Space to listen (when not typing)
    document.addEventListener('keydown', (e) => {
        if (e.code === 'Space' && document.activeElement !== textInput) {
            e.preventDefault();
            if (state === 'idle') {
                startListening();
            } else if (state === 'listening') {
                stopListening();
            }
        }
        // Escape to stop everything
        if (e.key === 'Escape') {
            synthesis.cancel();
            stopListening();
            setState('idle');
        }
    });
}

// ═══ Command helper for hints ═══
function execCmd(cmd) {
    transcript.textContent = cmd;
    processInput(cmd);
}
window.execCmd = execCmd;

// ═══ Pre-load voices ═══
if (synthesis.onvoiceschanged !== undefined) {
    synthesis.onvoiceschanged = () => { };
}
