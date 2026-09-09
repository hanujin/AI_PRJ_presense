const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');

// Exercise the real request handler with in-memory media and a mock model.
const source = ts.createSourceFile('dashboard.tsx', fs.readFileSync(path.join(__dirname, '../components/dashboard-shell.tsx'), 'utf8'), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
let handler;
function visit(node) {
  if (ts.isFunctionDeclaration(node) && node.name?.text === 'requestLiveFrame') handler = node.getText(source);
  ts.forEachChild(node, visit);
}
visit(source);
assert.ok(handler);
const js = ts.transpileModule(handler, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText;
function setup(respond = async () => ({ stressScore: 0.4, confidence: 0.6 }), overrides = {}) {
  const results = [], phases = [];
  const refs = { generation: { current: 0 }, inFlight: { current: false }, active: { current: false }, baseline: { current: null } };
  const env = {
    recordingActiveRef: refs.active, preparationFrameRef: refs.baseline,
    inferInFlightRef: refs.inFlight,
    analysisGenerationRef: refs.generation,
    inferenceAbortRef: { current: null },
    frameBufferRef: { current: Array.from({ length: 16 }, () => new Blob(['frame'])) },
    getRecentAudioPcm: () => new Float32Array(160000),
    setLiveStatus: () => {}, setPreparationPhase: (phase) => phases.push(phase),
    setModelConnected: () => {}, SIGNAL_SOURCE: 'live',
    inferMultimodalFrame: respond, applyFrame: (frame) => results.push(frame),
    ...overrides,
  };
  return { request: new Function(...Object.keys(env), `${js}; return requestLiveFrame;`)(...Object.values(env)), results, phases, refs };
}
test('preparation checks the model without adding any recorded results', async () => {
  const state = setup();
  await state.request(true);
  assert.deepEqual(state.phases, ['analyzing', 'ready']);
  assert.deepEqual(state.results, []);
  assert.equal(state.refs.baseline.current.stressScore, 0.4);
});
test('normal recording stores the current model result', async () => {
  const state = setup();
  await state.request();
  assert.equal(state.results.length, 1);
  assert.deepEqual(state.phases, []);
});
test('failed preparation does not enable completion or record a score', async () => {
  const state = setup(async () => { throw new Error('offline'); });
  await state.request(true);
  assert.deepEqual(state.phases, ['analyzing', 'error']);
  assert.deepEqual(state.results, []);
  assert.equal(state.refs.inFlight.current, false);
});
test('response from a discarded preparation is ignored', async () => {
  let resolve;
  const state = setup(() => new Promise((done) => { resolve = done; }));
  const pending = state.request(true);
  state.refs.generation.current++;
  resolve({ stressScore: 0.7, confidence: 0.7 });
  await pending;
  assert.deepEqual(state.phases, ['analyzing']);
  assert.deepEqual(state.results, []);
});

test('a partial audio window is not sent to inference', async () => {
  let requests = 0;
  const state = setup(async () => { requests++; }, {
    getRecentAudioPcm: () => new Float32Array(159000),
  });
  await state.request();
  assert.equal(requests, 0);
  assert.equal(state.refs.inFlight.current, false);
});

function loadFunction(name, env) {
  let found;
  function find(node) {
    if (ts.isFunctionDeclaration(node) && node.name?.text === name) found = node.getText(source);
    ts.forEachChild(node, find);
  }
  find(source);
  assert.ok(found);
  const code = ts.transpileModule(found, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText;
  return new Function(...Object.keys(env), `${code}; return ${name};`)(...Object.values(env));
}

test('video collection fills its 16-frame window in 10 seconds', async () => {
  let tick, interval;
  const frames = { current: [] };
  const start = loadFunction('startCollection', {
    startAudioMeter: () => {}, collectionTickRef: { current: null },
    analysisGenerationRef: { current: 0 }, collectionBusyRef: { current: false },
    captureFrameBlob: async () => new Blob(['frame']), frameBufferRef: frames,
    setCollectedFrames: () => {}, setLiveStatus: () => {},
    window: { setInterval: (fn, ms) => { tick = fn; interval = ms; return 1; } },
  });
  start({});
  for (let i = 0; i < 16; i++) await tick();
  assert.equal(frames.current.length, 16);
  assert.equal(interval * 16, 10000);
  await tick();
  assert.equal(frames.current.length, 16);
});

test('audio uses only the latest ten seconds and returns 16 kHz PCM', () => {
  const read = loadFunction('getRecentAudioPcm', {
    audioChunksRef: { current: [new Float32Array(4800).fill(1), new Float32Array(480000).fill(2)] },
    audioSampleRateRef: { current: 48000 },
  });
  const audio = read();
  assert.equal(audio.length, 160000);
  assert.ok(audio.every((sample) => sample === 2));
});

test('a preparation response finishing after start cannot replace the starting baseline or add a score', async () => {
  let resolve;
  const state = setup(() => new Promise((done) => { resolve = done; }));
  const baseline = { stressScore: 0.3, confidence: 0.8 };
  state.refs.baseline.current = baseline;
  const pending = state.request(true);
  state.refs.active.current = true;
  resolve({ stressScore: 0.9, confidence: 0.9 });
  await pending;
  assert.equal(state.refs.baseline.current, baseline);
  assert.deepEqual(state.results, []);
  assert.equal(state.refs.inFlight.current, false);
});

test('late inference from an ended session cannot enter the next session', async () => {
  let resolve;
  const state = setup(() => new Promise((done) => { resolve = done; }));
  const pending = state.request();
  state.refs.generation.current++;
  state.refs.active.current = true;
  resolve({ stressScore: 0.95, confidence: 0.9 });
  await pending;
  assert.deepEqual(state.results, []);
  assert.equal(state.refs.baseline.current, null);
});

test('stopping clears media and baseline, invalidates requests, and stops timers', () => {
  const controller = new AbortController();
  const env = {
    recordingActiveRef: { current: true }, preparationFrameRef: { current: { stressScore: 0.8 } },
    analysisGenerationRef: { current: 1 }, inferenceAbortRef: { current: controller },
    inferInFlightRef: { current: true }, audioChunksRef: { current: [new Float32Array(100)] },
    frameBufferRef: { current: [new Blob(['old'])] }, collectionTickRef: { current: 1 },
    collectionBusyRef: { current: true }, rafRef: { current: null }, tickRef: { current: 2 },
    audioContextRef: { current: null }, analyserRef: { current: {} }, audioProcessorRef: { current: {} },
    setCollectedFrames() {}, setAudioSeconds() {}, setVoiceDetected() {}, setAudioRunning() {}, setAudioLevel() {},
    window: { clearInterval() {} },
  };
  loadFunction('stopAnalysis', env)();
  assert.equal(controller.signal.aborted, true);
  assert.equal(env.analysisGenerationRef.current, 2);
  assert.equal(env.preparationFrameRef.current, null);
  assert.equal(env.recordingActiveRef.current, false);
  assert.deepEqual(env.audioChunksRef.current, []);
  assert.deepEqual(env.frameBufferRef.current, []);
  assert.equal(env.collectionTickRef.current, null);
  assert.equal(env.tickRef.current, null);
});
