import type { FindingType, ReviewStatus, Role, Severity } from "../domain/types";

export interface QaPredictionEvidenceDto {
  id: string;
  trackId: string;
  label: string;
  bbox: [number, number, number, number];
  confidence: number;
}

export interface QaCaseEvidenceDto {
  groundTruthBbox?: [number, number, number, number];
  groundTruthLabels?: RealDatasetLabelDto[];
  observedPredictions?: QaPredictionEvidenceDto[];
  imageUrl?: string;
  imageWidth?: number;
  imageHeight?: number;
  metrics?: Record<string, unknown>;
  evaluationId?: string;
  summary?: string;
  [key: string]: unknown;
}

export interface QaCaseDto {
  id: string;
  datasetId: string;
  datasetVersion: string;
  sourceSplit: string | null;
  sourceImageId: string | null;
  evaluationId: string | null;
  sequenceId: string;
  frameIndex: number;
  frameFileName: string;
  className: string;
  targetTrackId: string | null;
  errorType: FindingType;
  riskScore: number;
  priority: Severity;
  status: ReviewStatus;
  evidence: QaCaseEvidenceDto;
  recommendation: string;
  assignedTo: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface QaCaseListDto {
  count: number;
  results: QaCaseDto[];
  limit: number;
  offset: number;
}

export interface RealDatasetBBoxDto {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface RealDatasetLabelDto {
  id: string;
  className: string;
  /**
   * Tên class theo taxonomy YOLO/COCO để hiển thị. `null` khi class này YOLO không có
   * (UI ẩn nhãn đó). `className` vẫn là taxonomy gốc dùng cho annotation editor.
   */
  normalizedClassName?: string | null;
  bbox: RealDatasetBBoxDto;
  trackId?: string | null;
  attributes?: Record<string, boolean | number | string>;
}

export interface RealDatasetImageDto {
  id: string;
  split: string;
  dataset: string | null;
  release: string | null;
  filename: string;
  width: number;
  height: number;
  labelCount: number;
  labels: RealDatasetLabelDto[];
  imageUrl: string;
  frameSampleId: string | null;
  sequenceId: string | null;
  cameraChannel: string | null;
}

export interface RealDatasetImageListDto {
  count: number;
  results: RealDatasetImageDto[];
  split: string;
  dataset: string | null;
  limit: number;
  offset: number;
  availableSplits: string[];
  availableDatasets: string[];
  classes: string[];
  normalizedClasses?: string[];
}

export interface RealDatasetFrameSampleDto {
  id: string;
  sampleId: string;
  sequenceId: string;
  split: string;
  dataset: string | null;
  cameraCount: number;
  labelCount: number;
  cameras: RealDatasetImageDto[];
}

export interface RealDatasetFrameSampleListDto {
  count: number;
  imageCount: number;
  results: RealDatasetFrameSampleDto[];
  split: string;
  dataset: string | null;
  limit: number;
  offset: number;
  availableSplits: string[];
  availableDatasets: string[];
  classes: string[];
  normalizedClasses?: string[];
}

export interface RealDatasetPredictionDto {
  id: string;
  className: string;
  normalizedClassName?: string | null;
  bbox: RealDatasetBBoxDto;
  confidence: number;
}

export interface RealDatasetMatchDto {
  groundTruthId: string;
  predictionId: string;
  groundTruthClass: string;
  predictionClass: string;
  iou: number;
  classMatch: boolean;
}

export interface QaAgentIssueDto {
  labelId: string | null;
  issueType: string;
  severity: "high" | "medium" | "low";
  explanation: string;
  suggestedFix: string;
  evidence: Record<string, unknown>;
}

export interface QaAgentReportDto {
  imagePath: string;
  status: "pass" | "needs_review" | "error";
  summary: string;
  metrics: Record<string, unknown>;
  issues: QaAgentIssueDto[];
}

export interface RealDatasetEvaluationDto {
  evaluationId: string;
  datasetId: string;
  datasetVersion: string;
  modelName: string;
  image: RealDatasetImageDto;
  report: QaAgentReportDto;
  predictions: RealDatasetPredictionDto[];
  matches: RealDatasetMatchDto[];
  unmatchedGroundTruth: Record<string, unknown>[];
  unmatchedPredictions: Record<string, unknown>[];
  cached: boolean;
  persisted: boolean;
  createdCaseIds: string[];
  inferenceMode: "yolo";
}

export interface RealDatasetBatchEvaluationResultDto {
  imageId: string;
  evaluation: RealDatasetEvaluationDto | null;
  error: string | null;
}

export interface RealDatasetBatchEvaluationDto {
  count: number;
  succeeded: number;
  failed: number;
  inferenceBatchUsed: boolean;
  results: RealDatasetBatchEvaluationResultDto[];
}

export interface AnnotationDocumentDto {
  datasetId: string;
  datasetVersion: string;
  split: string;
  imageId: string;
  revision: number;
  image: RealDatasetImageDto;
  labels: RealDatasetLabelDto[];
  originalLabels: RealDatasetLabelDto[];
  updatedAt: string | null;
  updatedBy: string | null;
  changeNote: string | null;
}

export interface AnnotationRevisionSummaryDto {
  revision: number;
  labelCount: number;
  actorId: string | null;
  changeNote: string | null;
  createdAt: string | null;
}

export interface AnnotationRevisionListDto {
  count: number;
  results: AnnotationRevisionSummaryDto[];
}
export interface AuthenticatedUserDto {
  id: string;
  email: string;
  displayName: string;
  role: Role;
  disabled: boolean;
}

export interface AdminProjectDto {
  id: string;
  name: string;
  customerName: string;
  description?: string | null;
  status: string;
  createdBy: string;
  createdAt: string;
}

export interface AdminSubmissionDto {
  id: string;
  projectId: string;
  datasetType: string;
  sourceMethod: string;
  version: string;
  split?: string | null;
  status: string;
  sourcePrefix?: string | null;
  createdBy: string;
  createdAt: string;
  updatedAt: string;
}

export interface AdminUploadSessionDto {
  assetId: string;
  objectKey: string;
  uploadUrl: string | null;
  expiresIn: number;
}

export interface AdminBatchDto {
  id: string;
  projectId: string;
  name: string;
  instructions?: string | null;
  scopeJson: Record<string, unknown>;
  status: string;
  reviewerId?: string | null;
  createdBy: string;
  createdAt: string;
  updatedAt: string;
}

export interface AdminTeamHealthDto {
  generatedAt: string;
  totalTasks: number;
  byStage: Record<string, number>;
  annotatorWorkload: Record<string, { assigned: number; wip: number; approved: number; changesRequested: number }>;
  quality: { approvalRate: number | null; reworkRate: number | null };
  ranking: null;
}

export interface ApplicationUserListDto {
  count: number;
  results: AuthenticatedUserDto[];
}

export interface PipelineProgressDto {
  phase: string;
  percent: number;
  detail: string;
}

export interface PipelineLogDto {
  timestamp: string | null;
  message: string;
}

export interface PipelineEventDto {
  phase: string;
  status: string;
  message: string;
  createdAt: string;
}

export interface PipelineRunDto {
  runId: string;
  datasetType: string;
  release: string | null;
  split: string | null;
  status: string;
  requestedBy: string | null;
  createdAt: string | null;
  startedAt: string | null;
  finishedAt: string | null;
  batchJobId: string | null;
  requestGcsUri: string | null;
  canonicalPrefix: string | null;
  images: number;
  objects: number;
  stages: PipelineProgressDto[];
  events: PipelineEventDto[];
  logs: PipelineLogDto[];
  validation: Record<string, unknown> | null;
  result: Record<string, unknown> | null;
}

export interface PipelineRunListDto {
  count: number;
  results: PipelineRunDto[];
}
