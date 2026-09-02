import { useCallback, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useAuth } from "../../auth/AuthProvider";
import { useQaCasesQuery, useQaCaseStatusMutation, useRealDatasetFrameSamplesQuery } from "../../api/queries";
import { AuthenticatedImage } from "../../components/AuthenticatedImage";
import type { QaCaseDto } from "../../api/types";
import { Badge, Button, Card, StatusBadge } from "../../components/ui";
import type { FindingType, ReviewStatus, Severity } from "../../domain/types";
import { apiBoxIntersectsImage } from "../../utils/realDataset";
import { ApiQueueComparisonViewer } from "./components/ApiQueueComparisonViewer";
import { QueueAnalytics } from "./QueueAnalytics";
import {
  findingTypeLabels,
  priorityLabel,
  QueueKpiCard,
  QueuePageState,
  reviewStatuses,
  statusLabels,
} from "./queuePresentation";

type ApiQueueSortKey = "priority" | "risk" | "newest";

const completedStatuses = new Set<ReviewStatus>([
  "confirmed",
  "corrected",
  "rejected",
  "skipped",
]);
const priorityRank: Record<Severity, number> = {
  critical: 4,
  high: 3,
  medium: 2,
  low: 1,
};
const EMPTY_CASES: QaCaseDto[] = [];
const PAGE_SIZE = 10;
const DEFAULT_QA_QUEUE_DATASET = "nuscenes";
const DEFAULT_QA_QUEUE_SPLIT = "smoke";

function evidenceLines(qaCase: QaCaseDto): string[] {
  const lines: string[] = [];
  if (qaCase.evidence.groundTruthBbox) {
    const [x, y, width, height] = qaCase.evidence.groundTruthBbox;
    lines.push(`GT bbox: x=${x}, y=${y}, rộng=${width}, cao=${height}`);
  }
  const predictions = qaCase.evidence.observedPredictions ?? [];
  if (predictions.length === 0) {
    lines.push("Không có prediction tương ứng tại frame này.");
  } else {
    predictions.forEach((prediction) => {
      lines.push(
        `Prediction ${prediction.label} · độ tin cậy ${Math.round(prediction.confidence * 100)}% · track ${prediction.trackId}`,
      );
    });
  }
  return lines;
}

function caseTargetsImageContent(qaCase: QaCaseDto): boolean {
  if (!qaCase.targetTrackId) return true;
  const target = qaCase.evidence.groundTruthLabels?.find(
    (label) => label.id === qaCase.targetTrackId,
  );
  const width = qaCase.evidence.imageWidth;
  const height = qaCase.evidence.imageHeight;
  if (!target || !width || !height) return true;
  return apiBoxIntersectsImage(target.bbox, width, height);
}

export function ApiQAQueueView({
  onOpenEditor,
}: {
  onOpenEditor?: (split: string, imageId: string) => void;
}) {
  const [searchParameters, setSearchParameters] = useSearchParams();
  const auth = useAuth();
  const canReview = auth.user?.role === "reviewer" || auth.user?.role === "admin";
  const scopedDataset = searchParameters.get("dataset") ?? DEFAULT_QA_QUEUE_DATASET;
  const scopedSplit = searchParameters.get("split") ?? DEFAULT_QA_QUEUE_SPLIT;
  const scopedImageId = searchParameters.get("imageId") ?? undefined;
  const casesQuery = useQaCasesQuery({
    split: scopedSplit,
    datasetId: scopedDataset,
    sourceImageId: scopedImageId,
  });
  const rawCases = casesQuery.data?.results ?? EMPTY_CASES;
  const cases = useMemo(
    () => rawCases.filter(caseTargetsImageContent),
    [rawCases],
  );
  const error =
    casesQuery.error instanceof Error ? casesQuery.error.message : "";
  const [query, setQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<ReviewStatus | "all">("all");
  const [typeFilter, setTypeFilter] = useState<FindingType | "all">("all");
  const [sequenceFilter, setSequenceFilter] = useState("all");
  const [frameFilter, setFrameFilter] = useState("");
  const [classFilter, setClassFilter] = useState("all");
  const [minimumRisk, setMinimumRisk] = useState(0);
  const [sortBy, setSortBy] = useState<ApiQueueSortKey>("priority");
  const [selectedCaseId, setSelectedCaseId] = useState("");
  const [page, setPage] = useState(1);
  const [activeTab, setActiveTab] = useState<"cases" | "frames">("cases");
  const [selectedSequence, setSelectedSequence] = useState<string>("all");
  const [selectedFrameId, setSelectedFrameId] = useState<string>("");

  const allSamplesQuery = useRealDatasetFrameSamplesQuery(scopedSplit, 0, scopedDataset);
  const allSamples = allSamplesQuery.data?.results ?? [];

  const explorerSequences = useMemo(() => {
    return [...new Set(allSamples.map((sample) => sample.sequenceId))].sort();
  }, [allSamples]);

  const filteredExplorerSamples = useMemo(() => {
    if (selectedSequence === "all") return allSamples;
    return allSamples.filter((sample) => sample.sequenceId === selectedSequence);
  }, [allSamples, selectedSequence]);

  useEffect(() => {
    if (filteredExplorerSamples.length > 0) {
      const exists = filteredExplorerSamples.some((s) => s.id === selectedFrameId);
      if (!exists && filteredExplorerSamples[0]) {
        setSelectedFrameId(filteredExplorerSamples[0].id);
      }
    } else {
      setSelectedFrameId("");
    }
  }, [filteredExplorerSamples, selectedFrameId]);

  const activeSample = useMemo(() => {
    return filteredExplorerSamples.find((s) => s.id === selectedFrameId);
  }, [filteredExplorerSamples, selectedFrameId]);

  const [decisionMessage, setDecisionMessage] = useState("");

  const setDatasetScope = useCallback(
    (dataset: string) => {
      setSearchParameters({ dataset, split: DEFAULT_QA_QUEUE_SPLIT });
    },
    [setSearchParameters],
  );

  const sequenceOptions = useMemo(
    () => [...new Set(cases.map((qaCase) => qaCase.sequenceId))].sort(),
    [cases],
  );
  const classOptions = useMemo(
    () => [...new Set(cases.map((qaCase) => qaCase.className))].sort(),
    [cases],
  );
  const visibleCases = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    const normalizedFrame = frameFilter.trim();
    const filtered = cases.filter((qaCase) => {
      const searchable = [
        qaCase.id,
        qaCase.datasetId,
        qaCase.sequenceId,
        qaCase.frameFileName,
        qaCase.className,
        qaCase.targetTrackId ?? "",
      ]
        .join(" ")
        .toLowerCase();
      return (
        (!normalizedQuery || searchable.includes(normalizedQuery)) &&
        (statusFilter === "all" || qaCase.status === statusFilter) &&
        (typeFilter === "all" || qaCase.errorType === typeFilter) &&
        (sequenceFilter === "all" || qaCase.sequenceId === sequenceFilter) &&
        (!normalizedFrame ||
          String(qaCase.frameIndex).includes(normalizedFrame)) &&
        (classFilter === "all" || qaCase.className === classFilter) &&
        qaCase.riskScore >= minimumRisk
      );
    });

    return [...filtered].sort((left, right) => {
      if (sortBy === "newest") {
        return Date.parse(right.createdAt) - Date.parse(left.createdAt);
      }
      if (sortBy === "risk") return right.riskScore - left.riskScore;
      return (
        priorityRank[right.priority] - priorityRank[left.priority] ||
        right.riskScore - left.riskScore
      );
    });
  }, [
    cases,
    classFilter,
    frameFilter,
    minimumRisk,
    query,
    sequenceFilter,
    sortBy,
    statusFilter,
    typeFilter,
  ]);

  const pageCount = Math.max(1, Math.ceil(visibleCases.length / PAGE_SIZE));
  const pagedCases = visibleCases.slice(
    (page - 1) * PAGE_SIZE,
    page * PAGE_SIZE,
  );

  useEffect(() => {
    setPage(1);
  }, [
    classFilter,
    frameFilter,
    minimumRisk,
    query,
    sequenceFilter,
    sortBy,
    statusFilter,
    typeFilter,
  ]);

  useEffect(() => {
    setPage((current) => Math.min(current, pageCount));
  }, [pageCount]);

  useEffect(() => {
    if (!visibleCases.some((qaCase) => qaCase.id === selectedCaseId)) {
      setSelectedCaseId(visibleCases[0]?.id ?? "");
    }
  }, [selectedCaseId, visibleCases]);

  useEffect(() => {
    if (!casesQuery.isPending && cases.length === 0 && allSamples.length > 0) {
      setActiveTab("frames");
    }
  }, [allSamples.length, cases.length, casesQuery.isPending]);

  const selectedCase = visibleCases.find(
    (qaCase) => qaCase.id === selectedCaseId,
  );
  const statusMutation = useQaCaseStatusMutation();

  useEffect(() => {
    setDecisionMessage("");
  }, [selectedCaseId]);

  const canConfirm = Boolean(
    canReview &&
    selectedCase &&
    ["unreviewed", "in_review", "corrected"].includes(selectedCase.status),
  );

  const confirmSelectedCase = useCallback(async () => {
    if (!selectedCase || !canConfirm) return;
    setDecisionMessage("");
    try {
      await statusMutation.mutateAsync({
        caseId: selectedCase.id,
        status: "confirmed",
      });
      setDecisionMessage("Đã xác nhận case và ghi nhận quyết định review.");
    } catch (confirmError) {
      setDecisionMessage(
        confirmError instanceof Error
          ? confirmError.message
          : "Không thể xác nhận QA case.",
      );
    }
  }, [canConfirm, selectedCase, statusMutation]);

  const rejectSelectedCase = useCallback(async () => {
    if (!selectedCase || !canReview) return;
    setDecisionMessage("");
    try {
      await statusMutation.mutateAsync({
        caseId: selectedCase.id,
        status: "rejected",
      });
      setDecisionMessage("Đã bác bỏ finding và ghi nhận quyết định review.");
    } catch (decisionError) {
      setDecisionMessage(
        decisionError instanceof Error
          ? decisionError.message
          : "Không thể cập nhật QA case.",
      );
    }
  }, [canReview, selectedCase, statusMutation]);

  const reviewedCount = cases.filter((qaCase) =>
    completedStatuses.has(qaCase.status),
  ).length;
  const reviewProgress = Math.round(
    (reviewedCount / Math.max(cases.length, 1)) * 100,
  );
  const highRiskCount = cases.filter((qaCase) => qaCase.riskScore >= 80).length;
  const editableImages = new Set(
    cases.map((qaCase) => qaCase.sourceImageId).filter(Boolean),
  ).size;

  const errorDistribution = useMemo(
    () =>
      Object.entries(findingTypeLabels)
        .map(([type, label]) => ({
          type: type as FindingType,
          label,
          count: cases.filter((qaCase) => qaCase.errorType === type).length,
        }))
        .filter((item) => item.count > 0)
        .sort((left, right) => right.count - left.count),
    [cases],
  );
  const classDistribution = useMemo(
    () =>
      classOptions
        .map((label) => ({
          label,
          count: cases.filter((qaCase) => qaCase.className === label).length,
        }))
        .sort((left, right) => right.count - left.count),
    [cases, classOptions],
  );
  const clearFilters = useCallback(() => {
    setQuery("");
    setStatusFilter("all");
    setTypeFilter("all");
    setSequenceFilter("all");
    setFrameFilter("");
    setClassFilter("all");
    setMinimumRisk(0);
    setSortBy("priority");
    setPage(1);
  }, []);
  const clearImageScope = useCallback(() => {
    const dataset = searchParameters.get("dataset");
    setSearchParameters(dataset ? { dataset } : {}, { replace: true });
  }, [searchParameters, setSearchParameters]);

  if (casesQuery.isPending && allSamplesQuery.isPending && cases.length === 0) {
    return (
      <QueuePageState
        title="Đang tải QA Cases"
        detail="Đọc QA case và annotation revision từ PostgreSQL…"
      />
    );
  }

  if (error && cases.length === 0 && allSamples.length === 0) {
    return (
      <QueuePageState
        title="Không thể kết nối backend"
        detail={error}
        action={
          <Button variant="primary" onClick={() => void casesQuery.refetch()}>
            Thử lại
          </Button>
        }
        error
      />
    );
  }

  if (
    !casesQuery.isPending &&
    !allSamplesQuery.isPending &&
    cases.length === 0 &&
    allSamples.length === 0
  ) {
    return (
      <QueuePageState
        title={scopedImageId ? "Ảnh này chưa có QA Case" : "Chưa có QA Cases"}
        detail={
          scopedImageId
            ? "Frame đang chọn chưa tạo ra finding. Xem toàn bộ QA Cases hoặc chạy Agent cho frame này ở QA Queue."
            : "Chạy Agent ở QA Queue để phân tích ảnh và tạo QA Cases."
        }
        action={
          scopedImageId ? (
            <Button variant="secondary" onClick={clearImageScope}>
              Xem toàn bộ QA Cases
            </Button>
          ) : (
            <Button variant="secondary" onClick={() => void casesQuery.refetch()}>
              Tải lại
            </Button>
          )
        }
      />
    );
  }

  return (
    <div className="page-container queue-console-page">
      <header className="qa-cases-page-heading">
        <div>
          <span>Agent output</span>
          <h1>QA Cases</h1>
          <p>
            Danh sách các gợi ý đã được Agent phát hiện và lưu để reviewer xử
            lý.
          </p>
        </div>
      </header>
      {scopedImageId ? (
        <div className="queue-frame-scope" role="status">
          <div>
            <strong>QA cases của frame hiện tại</strong>
            <span>
              {scopedSplit ?? "dataset"} · {scopedImageId}
            </span>
          </div>
          <Button
            size="sm"
            variant="secondary"
            onClick={clearImageScope}
          >
            Xem toàn bộ QA cases
          </Button>
        </div>
      ) : null}
      <div className="queue-console-topline">
        <div className="queue-dataset-selector">
          <select
            value={scopedDataset}
            onChange={(event) => setDatasetScope(event.target.value)}
            aria-label="Chọn dataset"
          >
            <option value="nuscenes">nuScenes</option>
            <option value="kitti">KITTI</option>
          </select>
          <select
            value={scopedSplit}
            onChange={(event) => setSearchParameters({ dataset: scopedDataset, split: event.target.value })}
            aria-label="Chọn split"
          >
            <option value="product">product (Official)</option>
            <option value="smoke">smoke (Testing)</option>
          </select>
        </div>
        <label className="queue-global-search">
          <span aria-hidden="true">⌕</span>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Tìm case, sequence, frame, track, task hoặc job…"
            aria-label="Tìm kiếm trong QA Cases"
          />
          <kbd>/</kbd>
        </label>
      </div>

      {error ? (
        <div className="api-inline-warning" role="status">
          <span>Dữ liệu đang hiển thị có thể đã cũ: {error}</span>
          <Button
            size="sm"
            variant="secondary"
            onClick={() => void casesQuery.refetch()}
          >
            Tải lại
          </Button>
        </div>
      ) : null}

      <section className="queue-kpi-grid" aria-label="Tổng quan QA Cases">
        <QueueKpiCard
          icon="▣"
          label="QA cases trong DB"
          value={cases.length}
          detail={`${sequenceOptions.length} sequence đã seed`}
          tone="blue"
        />
        <QueueKpiCard
          icon="⚑"
          label="Case sau bộ lọc"
          value={visibleCases.length}
          detail={`${cases.length} case tổng cộng`}
          tone="red"
        />
        <QueueKpiCard
          icon="◎"
          label="Ảnh có thể chỉnh sửa"
          value={editableImages}
          detail="Lưu theo annotation revision"
          tone="purple"
        />
        <QueueKpiCard
          icon="✓"
          label="Đã review"
          value={reviewedCount}
          detail={`${reviewProgress}% của hàng đợi`}
          tone="green"
        />
        <QueueKpiCard
          icon="!"
          label="High-risk cases"
          value={highRiskCount}
          detail="Risk score từ 80 trở lên"
          tone="orange"
        />
      </section>

      <div className="queue-tabs-navigation">
        <button
          className={`queue-tab-button ${activeTab === "cases" ? "is-active" : ""}`}
          onClick={() => setActiveTab("cases")}
          type="button"
        >
          QA Cases Queue
        </button>
        <button
          className={`queue-tab-button ${activeTab === "frames" ? "is-active" : ""}`}
          onClick={() => setActiveTab("frames")}
          type="button"
        >
          All Dataset Frames
        </button>
      </div>

      {activeTab === "frames" ? (
        <section className="all-frames-explorer">
          <Card className="explorer-card">
            <div className="explorer-layout">
              {/* Sidebar */}
              <div className="explorer-sidebar">
                <label>
                  <span>Sequence (Scene)</span>
                  <select
                    value={selectedSequence}
                    onChange={(event) => {
                      setSelectedSequence(event.target.value);
                      setSelectedFrameId("");
                    }}
                  >
                    <option value="all">Tất cả sequence</option>
                    {explorerSequences.map((sequence) => (
                      <option key={sequence} value={sequence}>
                        {sequence}
                      </option>
                    ))}
                  </select>
                </label>

                <label>
                  <span>Chọn Frame ({filteredExplorerSamples.length})</span>
                  <div className="explorer-frames-list">
                    {filteredExplorerSamples.length > 0 ? (
                      filteredExplorerSamples.map((sample) => (
                        <button
                          key={sample.id}
                          className={`explorer-frame-item ${sample.id === selectedFrameId ? "is-active" : ""}`}
                          onClick={() => setSelectedFrameId(sample.id)}
                          type="button"
                        >
                          <strong>{sample.sampleId ? sample.sampleId.slice(0, 16) + "..." : sample.id}</strong>
                          <span>{sample.sequenceId} · {sample.cameraCount} views</span>
                        </button>
                      ))
                    ) : (
                      <div style={{ padding: "16px", color: "var(--text-muted, #748197)", fontSize: "11px", textAlign: "center" }}>
                        Không tìm thấy frame nào.
                      </div>
                    )}
                  </div>
                </label>
              </div>

              {/* Main Content: Camera Grid */}
              <div className="explorer-main-content">
                {activeSample ? (
                  <>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <h3>Frame views cho <code>{activeSample.sequenceId}</code></h3>
                      <Badge tone="info">{activeSample.cameras.length} camera góc</Badge>
                    </div>
                    <div className="camera-grid">
                      {activeSample.cameras.map((camera) => (
                        <div
                          key={camera.id}
                          className="camera-card"
                          onClick={() => onOpenEditor?.(scopedSplit, camera.id)}
                        >
                          <div className="camera-thumb">
                            <AuthenticatedImage sourcePath={camera.imageUrl} alt={camera.cameraChannel || "Camera view"} loading="lazy" />
                          </div>
                          <div className="camera-info">
                            <strong>{camera.cameraChannel ?? "Camera View"}</strong>
                            <span>{camera.width}x{camera.height}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                ) : (
                  <div className="empty-explorer-state">
                    <strong>Chưa chọn frame</strong>
                    <p>Vui lòng chọn một frame từ danh sách bên trái để xem các camera góc.</p>
                  </div>
                )}
              </div>
            </div>
          </Card>
        </section>
      ) : (
        <>
          <section className="queue-console-workbench">
        <Card className="queue-console-filter-panel">
          <div className="queue-panel-heading">
            <strong>Bộ lọc</strong>
            <button type="button" onClick={clearFilters}>
              ↻ Đặt lại
            </button>
          </div>
          <div className="queue-filter-stack">
            <label>
              <span>Dataset</span>
              <select
                value={scopedDataset}
                onChange={(event) => setDatasetScope(event.target.value)}
              >
                <option value="nuscenes">nuScenes</option>
                <option value="kitti">KITTI</option>
              </select>
            </label>
            <label>
              <span>Sequence</span>
              <select
                value={sequenceFilter}
                onChange={(event) => setSequenceFilter(event.target.value)}
              >
                <option value="all">Tất cả sequence</option>
                {sequenceOptions.map((sequence) => (
                  <option key={sequence} value={sequence}>
                    {sequence}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span>Frame</span>
              <input
                value={frameFilter}
                onChange={(event) => setFrameFilter(event.target.value)}
                placeholder="Nhập frame ID"
              />
            </label>
            <label>
              <span>Class</span>
              <select
                value={classFilter}
                onChange={(event) => setClassFilter(event.target.value)}
              >
                <option value="all">Tất cả</option>
                {classOptions.map((label) => (
                  <option key={label} value={label}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span>Loại lỗi</span>
              <select
                value={typeFilter}
                onChange={(event) =>
                  setTypeFilter(event.target.value as FindingType | "all")
                }
              >
                <option value="all">Tất cả</option>
                {Object.entries(findingTypeLabels).map(([type, label]) => (
                  <option key={type} value={type}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span>Trạng thái review</span>
              <select
                value={statusFilter}
                onChange={(event) =>
                  setStatusFilter(event.target.value as ReviewStatus | "all")
                }
              >
                <option value="all">Tất cả</option>
                {reviewStatuses.map((status) => (
                  <option key={status} value={status}>
                    {statusLabels[status]}
                  </option>
                ))}
              </select>
            </label>
            <label className="queue-risk-filter">
              <span>
                Risk score tối thiểu
              </span>
              <div className="queue-risk-input">
                <input
                  type="number"
                  min="0"
                  max="100"
                  step="1"
                  inputMode="numeric"
                  value={minimumRisk}
                  onChange={(event) =>
                    setMinimumRisk(Math.min(100, Math.max(0, Number(event.target.value) || 0)))
                  }
                />
                <span>/ 100</span>
              </div>
            </label>
            <label>
              <span>Sắp xếp theo</span>
              <select
                value={sortBy}
                onChange={(event) =>
                  setSortBy(event.target.value as ApiQueueSortKey)
                }
              >
                <option value="priority">Ưu tiên cao → thấp</option>
                <option value="risk">Risk giảm dần</option>
                <option value="newest">Mới nhất</option>
              </select>
            </label>
          </div>
        </Card>

        <Card className="queue-console-viewer-card">
          <ApiQueueComparisonViewer qaCase={selectedCase} />
        </Card>

        <Card className="queue-console-detail-panel">
          {selectedCase ? (
            <>
              <div className="queue-panel-heading queue-detail-heading">
                <div>
                  <strong>Chi tiết case</strong>
                  <small>{selectedCase.id}</small>
                </div>
                <Badge tone="info">API</Badge>
              </div>
              <dl className="queue-case-metadata">
                <div>
                  <dt>Nguồn</dt>
                  <dd>Dataset · {selectedCase.sourceSplit}</dd>
                </div>
                <div>
                  <dt>Image ID</dt>
                  <dd>{selectedCase.sourceImageId ?? "—"}</dd>
                </div>
                <div>
                  <dt>Sequence</dt>
                  <dd>{selectedCase.sequenceId}</dd>
                </div>
                <div>
                  <dt>Frame</dt>
                  <dd>
                    {selectedCase.frameIndex} · {selectedCase.frameFileName}
                  </dd>
                </div>
                <div>
                  <dt>Class</dt>
                  <dd>{selectedCase.className}</dd>
                </div>
                <div>
                  <dt>Error type</dt>
                  <dd>
                    <Badge tone={selectedCase.priority}>
                      {findingTypeLabels[selectedCase.errorType]}
                    </Badge>
                  </dd>
                </div>
                <div>
                  <dt>Risk score</dt>
                  <dd>
                    <Badge tone={selectedCase.priority}>
                      {selectedCase.riskScore} / 100
                    </Badge>
                  </dd>
                </div>
                <div>
                  <dt>Trạng thái</dt>
                  <dd>
                    <StatusBadge status={selectedCase.status} />
                  </dd>
                </div>
                <div>
                  <dt>Editor</dt>
                  <dd>
                    Revision{" "}
                    {String(selectedCase.evidence.annotationRevision ?? 0)}
                  </dd>
                </div>
              </dl>

              <div className="queue-detail-section">
                <strong>Bằng chứng</strong>
                <ul>
                  {evidenceLines(selectedCase).map((line) => (
                    <li key={line}>{line}</li>
                  ))}
                </ul>
              </div>

              <div className="queue-agent-explanation">
                <div>
                  <span>▣</span>
                  <strong>Giải thích của Agent</strong>
                </div>
                <p>
                  {selectedCase.evidence.summary ??
                    "Agent đã phát hiện sai lệch giữa GT và prediction tại frame được chọn."}
                </p>
              </div>

              <div className="queue-detail-section queue-recommendation-section">
                <strong>Đề xuất xử lý</strong>
                <p>{selectedCase.recommendation}</p>
              </div>

              <div className="queue-case-actions">
                <Button
                  variant="primary"
                  disabled={!canConfirm || statusMutation.isPending}
                  onClick={() => void confirmSelectedCase()}
                  title="Xác nhận annotation hiện tại"
                >
                  {statusMutation.isPending ? "Đang cập nhật…" : "✓ Xác nhận"}
                </Button>
                <Button
                  variant="secondary"
                  disabled={
                    !selectedCase.sourceSplit || !selectedCase.sourceImageId
                  }
                  onClick={() =>
                    selectedCase.sourceSplit &&
                    selectedCase.sourceImageId &&
                    onOpenEditor?.(
                      selectedCase.sourceSplit,
                      selectedCase.sourceImageId,
                    )
                  }
                  title="Mở đúng ảnh trong 2D Editor"
                >
                  ✎ Chỉnh sửa nhãn
                </Button>
                <Button
                  variant="danger"
                  disabled={!canReview || statusMutation.isPending}
                  onClick={() => void rejectSelectedCase()}
                >
                  × Bác bỏ
                </Button>
              </div>
              {decisionMessage ? (
                <div
                  className={
                    statusMutation.isError
                      ? "api-inline-warning"
                      : "api-status-message"
                  }
                  role={statusMutation.isError ? "alert" : "status"}
                >
                  {decisionMessage}
                </div>
              ) : null}
              <p className="api-phase-note">
                Editor lưu revision bất biến và tự cập nhật các QA case của cùng
                ảnh sang trạng thái đã sửa.
              </p>
            </>
          ) : (
            <div className="queue-detail-empty">
              Không có case phù hợp với bộ lọc hiện tại.
            </div>
          )}
        </Card>
      </section>

      <section className="queue-console-bottom-grid">
        <Card className="queue-console-table-card">
          <div className="queue-table-titlebar">
            <div>
              <strong>Danh sách QA cases</strong>
              <Badge tone="neutral">{visibleCases.length} items</Badge>
            </div>
            <button
              type="button"
              onClick={() => void casesQuery.refetch()}
              aria-label="Làm mới danh sách"
            >
              ↻
            </button>
          </div>
          <div className="queue-console-table-wrap">
            <table className="queue-console-table">
              <thead>
                <tr>
                  <th>
                    <input
                      type="checkbox"
                      disabled
                      aria-label="Chọn tất cả case"
                    />
                  </th>
                  <th>Case ID</th>
                  <th>Sequence</th>
                  <th>Frame</th>
                  <th>Class</th>
                  <th>Lỗi</th>
                  <th>Risk</th>
                  <th>Image ID</th>
                  <th>Trạng thái</th>
                  <th>Ưu tiên</th>
                </tr>
              </thead>
              <tbody>
                {pagedCases.map((qaCase) => (
                  <tr
                    className={
                      selectedCaseId === qaCase.id ? "is-selected" : ""
                    }
                    key={qaCase.id}
                    onClick={() => setSelectedCaseId(qaCase.id)}
                  >
                    <td>
                      <input
                        type="checkbox"
                        disabled
                        aria-label={`Chọn ${qaCase.id}`}
                      />
                    </td>
                    <td>
                      <button
                        type="button"
                        onClick={() => setSelectedCaseId(qaCase.id)}
                      >
                        {qaCase.id}
                      </button>
                    </td>
                    <td>{qaCase.sequenceId}</td>
                    <td>{qaCase.frameIndex}</td>
                    <td>{qaCase.className}</td>
                    <td>{findingTypeLabels[qaCase.errorType]}</td>
                    <td>
                      <Badge tone={qaCase.priority}>{qaCase.riskScore}</Badge>
                    </td>
                    <td>{qaCase.sourceImageId ?? "—"}</td>
                    <td>
                      <StatusBadge status={qaCase.status} />
                    </td>
                    <td>
                      <Badge tone={qaCase.priority}>
                        {priorityLabel(qaCase.priority)}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {visibleCases.length === 0 ? (
              <div className="queue-console-empty">
                <strong>Không có case phù hợp</strong>
                <span>Hãy giảm điều kiện lọc.</span>
                <Button size="sm" variant="secondary" onClick={clearFilters}>
                  Đặt lại bộ lọc
                </Button>
              </div>
            ) : null}
          </div>
          <div className="queue-table-pagination">
            <span>
              Hiển thị{" "}
              {visibleCases.length === 0 ? 0 : (page - 1) * PAGE_SIZE + 1} –{" "}
              {Math.min(page * PAGE_SIZE, visibleCases.length)} trong{" "}
              {visibleCases.length}
            </span>
            <label>
              <select value={PAGE_SIZE} disabled aria-label="Số case mỗi trang">
                <option value={PAGE_SIZE}>10 / trang</option>
              </select>
            </label>
            <div>
              <button
                type="button"
                disabled={page === 1}
                onClick={() => setPage((current) => Math.max(1, current - 1))}
              >
                ‹
              </button>
              {Array.from({ length: pageCount }, (_, index) => index + 1).map(
                (pageNumber) => (
                  <button
                    key={pageNumber}
                    type="button"
                    className={pageNumber === page ? "is-current" : ""}
                    onClick={() => setPage(pageNumber)}
                  >
                    {pageNumber}
                  </button>
                ),
              )}
              <button
                type="button"
                disabled={page === pageCount}
                onClick={() =>
                  setPage((current) => Math.min(pageCount, current + 1))
                }
              >
                ›
              </button>
            </div>
          </div>
        </Card>

        <QueueAnalytics
          errorDistribution={errorDistribution}
          classDistribution={classDistribution}
          totalCount={cases.length}
          reviewedCount={reviewedCount}
          reviewProgress={reviewProgress}
        />
      </section>
        </>
      )}
    </div>
  );
}
