import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const readSource = (relativePath: string) =>
  readFileSync(new URL(relativePath, import.meta.url), "utf8");

test("workspace workflow banner is removed", () => {
  const layoutSource = readSource("../src/components/layout.tsx");

  assert.doesNotMatch(layoutSource, /global-safety-banner/);
  assert.doesNotMatch(layoutSource, /AI proposes · human decides/);
});

test("application sidebar expands on hover and omits the API status card", () => {
  const layoutSource = readSource("../src/components/layout.tsx");
  const themeStyles = readSource("../src/styles/platform-theme.css");

  assert.doesNotMatch(layoutSource, /sidebarCollapsed/);
  assert.doesNotMatch(layoutSource, /sidebar-collapse-button/);
  assert.doesNotMatch(layoutSource, /sidebar-section-label/);
  assert.doesNotMatch(layoutSource, />Workspace</);
  assert.match(themeStyles, /\.app-shell-dark \.sidebar\s*{[\s\S]*?width: 68px;/);
  assert.match(themeStyles, /\.app-shell-dark \.sidebar:hover\s*{\s*width: 220px;/);
  assert.match(themeStyles, /\.app-shell-dark \.sidebar:hover\s*{[^}]*transition-delay: 70ms;/);
  assert.match(
    themeStyles,
    /\.app-shell-dark \.sidebar-nav-icon\s*{[^}]*left: 16px;[^}]*transform: none;/,
  );
  assert.doesNotMatch(themeStyles, /sidebar:hover \.sidebar-nav-icon/);
  assert.doesNotMatch(layoutSource, /API V1 \+ Supabase Auth/);
  assert.doesNotMatch(layoutSource, /API connected · review actions are audited\./);
  assert.doesNotMatch(layoutSource, /Private GCS dataset/);
});

test("all application scrollbars use semantic surface tokens", () => {
  const baseStyles = readSource("../src/styles/base.css");

  assert.match(baseStyles, /\*\s*{[^}]*scrollbar-width: thin;/);
  assert.match(baseStyles, /scrollbar-color: var\(--color-surface-3\) var\(--color-canvas\);/);
  assert.match(
    baseStyles,
    /\*::-webkit-scrollbar-thumb\s*{[^}]*background: var\(--color-surface-3\);/,
  );
});

test("QA Cases omits implementation and reviewer context chips", () => {
  const apiQueueSource = readSource("../src/features/qa-queue/ApiQAQueueView.tsx");

  assert.doesNotMatch(apiQueueSource, /FastAPI · Built-in Editor/);
  assert.doesNotMatch(apiQueueSource, /QA Reviewer/);
  assert.doesNotMatch(apiQueueSource, /Agent chỉ đề xuất, không tự động sửa nhãn/);
  assert.doesNotMatch(apiQueueSource, /queue-context-chips/);
});

test("QA comparison viewers pan by pointer drag instead of scrollbars", () => {
  const apiViewerSource = readSource(
    "../src/features/qa-queue/components/ApiQueueComparisonViewer.tsx",
  );
  const mockViewerSource = readSource(
    "../src/features/qa-queue/components/MockQueueComparisonViewer.tsx",
  );
  const queueStyles = readSource("../src/styles/queue-console.css");

  for (const source of [apiViewerSource, mockViewerSource]) {
    assert.match(source, /onPointerDown={startPan}/);
    assert.match(source, /setPointerCapture\(event\.pointerId\)/);
    assert.match(source, /translate\(\$\{pan\.x\}px, \$\{pan\.y\}px\)/);
  }
  assert.match(queueStyles, /\.queue-viewer-stage\s*{[^}]*overflow: hidden;/);
  assert.match(queueStyles, /\.queue-viewer-stage\.is-dragging\s*{\s*cursor: grabbing;/);
});

test("QA case risk filters use bounded numeric inputs", () => {
  const apiQueueSource = readSource("../src/features/qa-queue/ApiQAQueueView.tsx");
  const mockQueueSource = readSource("../src/features/qa-queue/MockQAQueueView.tsx");

  for (const source of [apiQueueSource, mockQueueSource]) {
    assert.match(source, /className="queue-risk-input"/);
    assert.match(source, /type="number"/);
    assert.doesNotMatch(source, /className="queue-risk-filter"[\s\S]{0,300}type="range"/);
  }
});

test("API dataset filter updates the URL-scoped dataset", () => {
  const apiQueueSource = readSource("../src/features/qa-queue/ApiQAQueueView.tsx");

  assert.match(apiQueueSource, /value={scopedDataset}/);
  assert.match(apiQueueSource, /DEFAULT_QA_QUEUE_SPLIT = "smoke"/);
  assert.match(apiQueueSource, /setDatasetScope\(event\.target\.value\)/);
  assert.doesNotMatch(apiQueueSource, /split: "product"/);
  assert.doesNotMatch(apiQueueSource, /<select value="active" disabled>/);
});

test("Agent explanation uses a black surface with white text", () => {
  const queueStyles = readSource("../src/styles/queue-console.css");

  assert.match(
    queueStyles,
    /\.queue-agent-explanation\s*{[\s\S]*?color: #fff;[\s\S]*?background: #080b10;/,
  );
});

test("editor synchronizes object and Agent suggestion selection", () => {
  const editorSource = readSource("../src/views/AnnotatorWorkspaceView.tsx");
  const editorStyles = readSource("../src/styles/label-editor.css");

  assert.match(editorSource, /const synchronizedSuggestionId = useMemo/);
  assert.match(editorSource, /onClick=\{\(\) => selectObject\(object\.id\)\}/);
  assert.match(editorSource, /suggestion\.id === synchronizedSuggestionId/);
  assert.doesNotMatch(editorSource, /agentHighlightedObjectId/);
  assert.match(
    editorStyles,
    /\.editor-agent-suggestions-list > button\.is-selected\s*{[\s\S]*?border: 2px solid #ff9f1c;[\s\S]*?background: transparent;/,
  );
  assert.match(
    editorStyles,
    /\.editor-object-list > button\.is-selected\s*{[\s\S]*?border: 2px solid #ff9f1c;[\s\S]*?background: transparent;/,
  );
});
