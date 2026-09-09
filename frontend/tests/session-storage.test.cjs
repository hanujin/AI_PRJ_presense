const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');

const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, '../lib/supabase/data.ts'), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText;

function setup(options = {}) {
  const events = [];
  const row = { id: 'record', title: 'Practice', session_date: '2026-09-09', duration_seconds: 30,
    result: 'Stable', stress_average: 0.4, diagnosis: 'Analysis', timeline: [20, 40], video_path: 'user/video.webm' };
  const client = {
    auth: { getUser: async () => ({ data: { user: { id: 'user' } }, error: null }) },
    storage: { from: () => ({
      upload: async (name) => { events.push(['upload', name]); return { error: options.uploadError }; },
      remove: async (names) => { events.push(['remove', names]); return { error: null }; },
      createSignedUrl: async () => ({ data: options.signError ? null : { signedUrl: 'https://example.test/video' }, error: options.signError }),
    }) },
    from: (table) => ({
      insert: async (value) => {
        events.push([table, value]);
        if (table === 'diagnosis_history' && options.summaryThrows) throw new Error('offline');
        return { error: table === 'presentation_records' ? options.insertError : null };
      },
      select: () => ({ order: async () => ({ data: [row], error: null }) }),
    }),
  };
  const module = { exports: {} };
  new Function('require', 'module', 'exports', code)((name) => {
    if (name === '@/lib/supabase/client') return { getSupabaseBrowserClient: () => client };
    if (name === '@/lib/app-data') return {};
    throw new Error(`Unexpected import: ${name}`);
  }, module, module.exports);
  return { api: module.exports, events };
}
const input = { title: 'Practice', durationSeconds: 30, averageStress: 0.4, result: 'Stable',
  diagnosis: 'Analysis', nextAction: 'Practice', sceneLabel: 'Camera', timeline: [20, 40],
  video: new Blob(['test recording'], { type: 'video/webm' }) };

test('video uploads before the session is inserted with its video path and timeline', async () => {
  const { api, events } = setup();
  await api.savePracticeSession(input);
  assert.equal(events[0][0], 'upload');
  assert.equal(events[1][0], 'presentation_records');
  assert.equal(events[1][1].video_path, events[0][1]);
  assert.deepEqual(events[1][1].timeline, input.timeline);
});
test('failed upload does not create a misleading video-less session', async () => {
  const { api, events } = setup({ uploadError: { message: 'Bucket not found' } });
  await assert.rejects(api.savePracticeSession(input), /Bucket not found/);
  assert.equal(events.length, 1);
});
test('failed session insert removes only its newly uploaded video', async () => {
  const { api, events } = setup({ insertError: { message: 'missing column' } });
  await assert.rejects(api.savePracticeSession(input), /missing column/);
  assert.deepEqual(events[2], ['remove', [events[0][1]]]);
});
test('analysis-only save does not upload video', async () => {
  const { api, events } = setup();
  await api.savePracticeSession({ ...input, video: null });
  assert.equal(events[0][0], 'presentation_records');
  assert.equal(events[0][1].video_path, null);
});
test('summary failure after persistence is a warning, not a failed save', async () => {
  const { api } = setup({ summaryThrows: true });
  const result = await api.savePracticeSession(input);
  assert.match(result.warning, /offline/);
});
test('history retains a saved video path and explains signed URL failure', async () => {
  const { api } = setup({ signError: { message: 'Access denied' } });
  const [record] = await api.loadSessionHistory();
  assert.equal(record.videoPath, 'user/video.webm');
  assert.equal(record.videoUrl, null);
  assert.equal(record.videoError, 'Access denied');
  assert.deepEqual(record.timeline, [20, 40]);
});
