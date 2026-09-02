import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
  useEvaluateRealDatasetImageMutation,
  useRealDatasetFrameSamplesQuery,
} from "../api/queries";
import type {
  RealDatasetEvaluationDto,
  RealDatasetImageDto,
  RealDatasetLabelDto,
  RealDatasetPredictionDto,
} from "../api/types";
import {
  AuthenticatedImage,
  useAuthenticatedAssetUrl,
} from "../components/AuthenticatedImage";
import { labelGuardianApiV1 } from "../api/labelGuardianApi";
import { Badge, Button, Card, SectionHeading } from "../components/ui";
import {
  apiBoxIntersectsImage,
  reportForSelectedImage,
} from "../utils/realDataset";

const PAGE_SIZE = 10;
const boxColors = [
  "#2563eb",
  "#dc2626",
  "#16a34a",
  "#9333ea",
  "#ea580c",
  "#0891b2",
  "#ca8a04",
  "#475569",
];

function colorForLabel(label: string): string {
  const index =
    [...label].reduce((sum, character) => sum + character.charCodeAt(0), 0) %
    boxColors.length;
  return boxColors[index];
}

// Tô nổi bật toạ độ ("x 809–964, y 182–328 (theo pixel)") và độ tin cậy ("58%")
// trong text giải thích / đề xuất của agent.
const AGENT_HIGHLIGHT_PATTERN =
  /(\d+(?:[.,]\d+)?\s?%|x:?\s?\d+\s?[–-]\s?\d+,\s?y:?\s?\d+\s?[–-]\s?\d+(?:\s?\(theo pixel\))?)/g;

function highlightAgentText(text: string): Array<string | JSX.Element> {
  return text.split(AGENT_HIGHLIGHT_PATTERN).map((part, index) => {
    if (index % 2 === 0) return part;
    const isConfidence = /%\s*$/.test(part);
    return (
      <span
        key={index}
        className={`agent-hl ${isConfidence ? "agent-hl-conf" : "agent-hl-coord"}`}
      >
        {part.trim()}
      </span>
    );
  });
}

// Nhãn có class không tồn tại bên YOLO/COCO -> không hiển thị trên UI.
function isDisplayableLabel(label: RealDatasetLabelDto): boolean {
  return Boolean(label.normalizedClassName);
}

function displayedLabelCount(image: RealDatasetImageDto): number {
  return image.labels.filter(
    (label) =>
      isDisplayableLabel(label) &&
      apiBoxIntersectsImage(label.bbox, image.width, image.height),
  ).length;
}

function evaluationKeyForImage(image: RealDatasetImageDto): string {
  return [
    image.dataset ?? "dataset",
    image.release ?? "release",
    image.split,
    image.id,
  ].join(":");
}

function evaluationKeyForResult(evaluation: RealDatasetEvaluationDto): string {
  return [
    evaluation.datasetId,
    evaluation.datasetVersion,
    evaluation.image.split,
    evaluation.image.id,
  ].join(":");
}

function AnnotationBox({ label }: { label: RealDatasetLabelDto }) {
  const { x1, y1, x2, y2 } = label.bbox;
  const shownClass = label.normalizedClassName;
  if (!shownClass) return null;
  const color = colorForLabel(shownClass);
  return (
    <g>
      <rect
        x={x1}
        y={y1}
        width={x2 - x1}
        height={y2 - y1}
        fill="none"
        stroke={color}
        strokeWidth="3"
      />
      <rect
        x={x1}
        y={Math.max(0, y1 - 22)}
        width={Math.max(64, shownClass.length * 9)}
        height="22"
        fill={color}
      />
      <text
        x={x1 + 5}
        y={Math.max(15, y1 - 6)}
        fill="white"
        fontSize="14"
        fontWeight="700"
      >
        {shownClass}
      </text>
    </g>
  );
}

function PredictionBox({
  prediction,
}: {
  prediction: RealDatasetPredictionDto;
}) {
  const { x1, y1, x2, y2 } = prediction.bbox;
  const shownClass = prediction.normalizedClassName ?? prediction.className;
  return (
    <g>
      <rect
        x={x1}
        y={y1}
        width={x2 - x1}
        height={y2 - y1}
        fill="none"
        stroke="#f59e0b"
        strokeWidth="3"
        strokeDasharray="9 5"
      />
      <rect
        x={x1}
        y={Math.max(0, y1 - 22)}
        width={Math.max(90, shownClass.length * 9)}
        height="22"
        fill="#f59e0b"
      />
      <text
        x={x1 + 5}
        y={Math.max(15, y1 - 6)}
        fill="#111827"
        fontSize="14"
        fontWeight="700"
      >
        {shownClass} · {Math.round(prediction.confidence * 100)}%
      </text>
    </g>
  );
}

export function RealDataQAView() {
  const lang = (localStorage.getItem("label-guardian-lang") as "en" | "vi") || "en";
  const t = (en: string, vi: string) => (lang === "en" ? en : vi);

  const [searchParams, setSearchParams] = useSearchParams();
  const selectedDataset = searchParams.get("dataset") || "nuscenes";
  const requestedSplit = searchParams.get("split") || import.meta.env.VITE_DATASET_DEFAULT_SPLIT || "product";
  const [selectedSequence, setSelectedSequence] = useState<string>("all");
  const [offset, setOffset] = useState(0);
  const [selectedId, setSelectedId] = useState<string>();
  const [evaluationsByImageKey, setEvaluationsByImageKey] = useState<
    Record<string, RealDatasetEvaluationDto>
  >({});
  const [comparisonMode, setComparisonMode] = useState<
    "gt" | "prediction" | "both"
  >("both");

  // Metadata query (with high limit) just to extract available sequences
  const metadataQuery = useRealDatasetFrameSamplesQuery(
    requestedSplit,
    0,
    selectedDataset,
    undefined,
    200,
  );

  const availableSequences = useMemo(() => {
    const results = metadataQuery.data?.results ?? [];
    return [...new Set(results.map((sample) => sample.sequenceId))].sort();
  }, [metadataQuery.data]);

  // Paginated query for the active list (loads only PAGE_SIZE = 10 samples)
  const samplesQuery = useRealDatasetFrameSamplesQuery(
    requestedSplit,
    offset,
    selectedDataset,
    selectedSequence === "all" ? undefined : selectedSequence,
    PAGE_SIZE,
  );
  const split = requestedSplit ?? samplesQuery.data?.split ?? "";
  const evaluation = useEvaluateRealDatasetImageMutation();
  const samples = samplesQuery.data?.results ?? [];
  const images = useMemo(
    () => samples.flatMap((sample) => sample.cameras),
    [samples],
  );

  useEffect(() => {
    if (!evaluation.data) return;
    setEvaluationsByImageKey((previous) => ({
      ...previous,
      [evaluationKeyForResult(evaluation.data)]: evaluation.data,
    }));
  }, [evaluation.data]);

  useEffect(() => {
    if (!images.some((image) => image.id === selectedId)) {
      setSelectedId(images[0]?.id);
    }
  }, [images, selectedId]);

  useEffect(() => {
    if (document.hidden || !selectedId) return;
    const index = images.findIndex((img) => img.id === selectedId);
    if (index === -1) return;

    const connection = (navigator as any).connection;
    if (connection && (connection.saveData || connection.effectiveType === "slow-2g" || connection.effectiveType === "2g")) {
      return;
    }

    const prefetchImage = (img?: RealDatasetImageDto) => {
      if (!img) return;
      labelGuardianApiV1.fetchAsset(img.imageUrl).catch(() => {});
    };

    prefetchImage(images[index + 1]);
    prefetchImage(images[index - 1]);
  }, [selectedId, images]);

  const selected = useMemo(
    () => images.find((image) => image.id === selectedId) ?? images[0],
    [images, selectedId],
  );
  const selectedAsset = useAuthenticatedAssetUrl(selected?.imageUrl);
  const selectedSample = useMemo(
    () =>
      samples.find((sample) =>
        sample.cameras.some((camera) => camera.id === selected?.id),
      ) ?? samples[0],
    [samples, selected?.id],
  );
  const displayedLabels = useMemo(
    () =>
      selected?.labels.filter(
        (label) =>
          isDisplayableLabel(label) &&
          apiBoxIntersectsImage(label.bbox, selected.width, selected.height),
      ) ?? [],
    [selected],
  );
  const selectedEvaluation = selected
    ? evaluationsByImageKey[evaluationKeyForImage(selected)]
    : undefined;
  const selectedPredictions = selectedEvaluation?.predictions ?? [];
  const report = reportForSelectedImage(selectedEvaluation, selected?.id);
  const lastPage = samplesQuery.data ? offset + PAGE_SIZE >= samplesQuery.data.count : true;
  const datasetOptions = samplesQuery.data?.availableDatasets.length
    ? samplesQuery.data.availableDatasets
    : ["nuscenes", "kitti"];

  const updateUrlFilter = (key: "dataset" | "split", value: string) => {
    const next = new URLSearchParams(searchParams);
    next.set(key, value);
    setSearchParams(next, { replace: true });
  };

  if (samplesQuery.isPending) {
    return (
      <Card className="real-data-state">
        <strong>{t("Loading dataset...", "Đang nạp dataset...")}</strong>
        <p>{t("Fetching dataset metadata and annotations from database.", "Đang nạp dữ liệu metadata và nhãn đối tượng.")}</p>
      </Card>
    );
  }
  if (samplesQuery.isError) {
    return (
      <Card className="real-data-state is-error">
        <strong>{t("Failed to load dataset", "Không thể tải tập dữ liệu")}</strong>
        <p>{samplesQuery.error.message}</p>
        <code>
          DATASET_BACKEND=database · split={requestedSplit ?? "backend default"}
        </code>
      </Card>
    );
  }

  return (
    <div className="real-data-page">
      <div className="real-data-heading">
        <SectionHeading
          eyebrow={t("QA Triage", "Phân loại QA")}
          title={t("Triage Frame & Generate QA Cases", "Triage Frame và tạo QA Case")}
          description={t("Access and triage sensor frames to execute validation runs and log QA findings.", "Truy xuất và triage các sensor frame để thực thi tiến trình đánh giá và ghi nhận lỗi QA.")}
        />
        <div className="real-data-controls">
          <label>
            <span>Dataset</span>
            <select
              value={selectedDataset}
              onChange={(event) => {
                updateUrlFilter("dataset", event.target.value);
                setOffset(0);
                setSelectedId(undefined);
                setSelectedSequence("all");
                evaluation.reset();
              }}
            >
              {datasetOptions.map((item) => <option key={item} value={item}>{item === "nuscenes" ? "nuScenes" : item.toUpperCase()}</option>)}
            </select>
          </label>
          <label>
            <span>Split</span>
            <select
              value={split}
              onChange={(event) => {
                updateUrlFilter("split", event.target.value);
                setOffset(0);
                setSelectedId(undefined);
                setSelectedSequence("all");
                evaluation.reset();
              }}
            >
              {samplesQuery.data?.availableSplits.map((item) => (
                <option key={item}>{item}</option>
              ))}
            </select>
          </label>
          <label>
            <span>Scene (Sequence)</span>
            <select
              value={selectedSequence}
              onChange={(event) => {
                setSelectedSequence(event.target.value);
                setOffset(0);
                setSelectedId(undefined);
                evaluation.reset();
              }}
            >
              <option value="all">Tất cả sequence</option>
              {availableSequences.map((seq) => (
                <option key={seq} value={seq}>{seq}</option>
              ))}
            </select>
          </label>
          <Badge tone="info">
            {samplesQuery.data?.count.toLocaleString()} frame samples
          </Badge>
          <Badge tone="neutral">
            {samplesQuery.data?.imageCount.toLocaleString()} camera views
          </Badge>
          <Badge tone="neutral">
            {(samplesQuery.data?.normalizedClasses ?? samplesQuery.data?.classes)?.length} lớp
          </Badge>
        </div>
      </div>

      <div className="real-data-workspace">
        <Card className="real-data-browser">
          <div className="real-data-browser-header">
            <strong>
              Samples {offset + 1}–
              {Math.min(offset + PAGE_SIZE, samplesQuery.data?.count ?? 0)}
            </strong>
            <span>{split}</span>
          </div>
          <div className="real-data-samples">
            {samples.map((sample, sampleIndex) => (
              <section
                key={sample.id}
                className={
                  sample.id === selectedSample?.id ? "is-selected" : ""
                }
              >
                <button
                  type="button"
                  className="real-data-sample-heading"
                  onClick={() => {
                    setSelectedId(sample.cameras[0]?.id);
                    evaluation.reset();
                  }}
                >
                  <span>
                    <strong>Frame {offset + sampleIndex + 1}</strong>
                    <small>{sample.sequenceId}</small>
                  </span>
                  <span>
                    <strong>{sample.cameraCount} cameras</strong>
                    <small>
                      {sample.cameras.reduce(
                        (total, camera) => total + displayedLabelCount(camera),
                        0,
                      )}{" "}
                      labels displayed
                    </small>
                  </span>
                  <code>{sample.sampleId}</code>
                </button>
                <div className="real-data-thumbnails">
                  {sample.cameras.map((image) => (
                    <button
                      key={image.id}
                      type="button"
                      className={image.id === selected?.id ? "is-selected" : ""}
                      onClick={() => {
                        setSelectedId(image.id);
                        evaluation.reset();
                      }}
                    >
                      <AuthenticatedImage
                        sourcePath={`${image.imageUrl}?size=thumbnail`}
                        alt={`${sample.sampleId} ${image.cameraChannel ?? image.id}`}
                        loading="lazy"
                        // @ts-ignore
                        fetchpriority="low"
                      />
                      <span>
                        <strong>
                          {image.cameraChannel?.replace("CAM_", "") ?? image.id}
                        </strong>
                        <small>
                          {displayedLabelCount(image)}/{image.labelCount} labels
                        </small>
                      </span>
                    </button>
                  ))}
                </div>
              </section>
            ))}
          </div>
          <div className="real-data-pagination">
            <Button
              size="sm"
              disabled={offset === 0}
              onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
            >
              Trang trước
            </Button>
            <span>{Math.floor(offset / PAGE_SIZE) + 1} / {Math.ceil((samplesQuery.data?.count ?? 0) / PAGE_SIZE) || 1}</span>
            <Button
              size="sm"
              disabled={lastPage}
              onClick={() => setOffset(offset + PAGE_SIZE)}
            >
              Trang sau
            </Button>
          </div>
        </Card>

        <div className="real-data-main">
          <Card className="real-data-viewer">
            {selected ? (
              <>
                <div className="real-data-viewer-header">
                  <div>
                    <span className="eyebrow">
                      {selectedSample?.sequenceId} / {selectedSample?.sampleId}
                    </span>
                    <h2>{selected.cameraChannel ?? selected.filename}</h2>
                  </div>
                  <div>
                    <div
                      className="compare-segmented"
                      role="group"
                      aria-label="Chế độ so sánh"
                    >
                      <button
                        className={comparisonMode === "gt" ? "is-active" : ""}
                        type="button"
                        onClick={() => setComparisonMode("gt")}
                      >
                        GT
                      </button>
                      <button
                        className={
                          comparisonMode === "prediction" ? "is-active" : ""
                        }
                        type="button"
                        onClick={() => setComparisonMode("prediction")}
                      >
                        YOLO
                      </button>
                      <button
                        className={comparisonMode === "both" ? "is-active" : ""}
                        type="button"
                        onClick={() => setComparisonMode("both")}
                      >
                        Cả hai
                      </button>
                    </div>
                    <Badge tone="success">
                      GT · {displayedLabels.length}/{selected.labelCount}
                    </Badge>
                    <Badge tone="info">
                      YOLO · {selectedPredictions.length}
                    </Badge>
                    <span>
                      {selected.width} × {selected.height}
                    </span>
                  </div>
                </div>
                <svg
                  viewBox={`0 0 ${selected.width} ${selected.height}`}
                  role="img"
                  aria-label={`Ground truth frame ${selected.id}`}
                >
                  <image
                    href={selectedAsset.source}
                    width={selected.width}
                    height={selected.height}
                    // @ts-ignore
                    fetchpriority="high"
                  />
                  {comparisonMode !== "prediction"
                    ? displayedLabels.map((label) => (
                        <AnnotationBox key={label.id} label={label} />
                      ))
                    : null}
                  {comparisonMode !== "gt"
                    ? selectedPredictions.map((prediction) => (
                        <PredictionBox
                          key={prediction.id}
                          prediction={prediction}
                        />
                      ))
                    : null}
                </svg>
                <div className="real-data-legend">
                  <span>
                    <i style={{ background: "#2563eb" }} />
                    Ground Truth
                  </span>
                  <span>
                    <i style={{ background: "#f59e0b" }} />
                    YOLO prediction
                  </span>
                </div>
              </>
            ) : (
              <p>Split này chưa có ảnh.</p>
            )}
          </Card>

          <Card className="real-data-agent">
            <div className="real-data-agent-header">
              <div>
                <span className="eyebrow">LangGraph + YOLO</span>
                <h2>Label QA Agent</h2>
              </div>
              <Button
                variant="primary"
                disabled={!selected || evaluation.isPending}
                onClick={() =>
                  selected &&
                  evaluation.mutate({
                    split,
                    imageId: selected.id,
                    persist: true,
                  })
                }
              >
                {evaluation.isPending
                  ? t("Running verification...", "Đang chạy kiểm tra...")
                  : t("Run Agent & Generate QA Cases", "Chạy Agent & tạo QA Cases")}
              </Button>
            </div>
            {evaluation.isError ? (
              <div className="real-data-agent-error">
                <strong>{t("Verification request failed", "Yêu cầu kiểm tra thất bại")}</strong>
                <p>{evaluation.error.message}</p>
              </div>
            ) : null}
            {!report && !evaluation.isPending ? (
              <div className="real-data-agent-empty">
                <p>
                  {t("Execute the validation agent on the selected frame to detect annotations defects.", "Chạy bộ máy kiểm tra trên frame đang chọn để phát hiện các khuyết tật nhãn.")}
                </p>
                <small>
                  {t("Running deep validation using active perception models and consistency rule sets.", "Chạy kiểm thử chuyên sâu bằng mô hình nhận diện và bộ quy tắc kiểm tra nhất quán.")}
                </small>
              </div>
            ) : null}
            {report ? (
              <div className={`real-data-report is-${report.status}`}>
                <div className="real-data-report-summary">
                  <Badge
                    tone={
                      report.status === "pass"
                        ? "success"
                        : report.status === "error"
                          ? "high"
                          : "info"
                    }
                  >
                    {report.status}
                  </Badge>
                  {selectedEvaluation?.cached ? (
                    <span>Cached</span>
                  ) : (
                    <span>Fresh inference</span>
                  )}
                  {selectedEvaluation?.persisted ? (
                    <span>
                      {t("Persisted", "Đã lưu")} · {selectedEvaluation.createdCaseIds.length} QA cases
                    </span>
                  ) : null}
                </div>
                <p>{report.summary}</p>
                <div className="real-data-metrics">
                  {Object.entries(report.metrics).map(([key, value]) => (
                    <div key={key}>
                      <span>{key.replaceAll("_", " ")}</span>
                      <strong>{String(value)}</strong>
                    </div>
                  ))}
                </div>
                <div className="real-data-issues">
                  {report.issues.map((issue, index) => (
                    <article key={`${issue.issueType}-${index}`}>
                      <div>
                        <Badge tone={issue.severity}>{issue.severity}</Badge>
                        <strong>{issue.issueType.replaceAll("_", " ")}</strong>
                      </div>
                      <p>{highlightAgentText(issue.explanation)}</p>
                      {issue.suggestedFix ? (
                        <small className="agent-suggestion">
                          <b>{t("Suggested fix", "Đề xuất")}:</b>{" "}
                          {highlightAgentText(issue.suggestedFix)}
                        </small>
                      ) : null}
                    </article>
                  ))}
                </div>
              </div>
            ) : null}
          </Card>
        </div>
      </div>
    </div>
  );
}
