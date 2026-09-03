import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { labelGuardianApiV1 } from "./labelGuardianApi";
import type { RealDatasetLabelDto } from "./types";

export const apiQueryKeys = {
  qaCases: ["api-v1", "qa-cases"] as const,
  qaCasesForImage: (split?: string, imageId?: string) =>
    ["api-v1", "qa-cases", split, imageId] as const,
  realDatasetImages: (split: string | undefined, dataset: string | undefined, offset: number) =>
    ["api-v1", "dataset", "images", split, dataset, offset] as const,
  realDatasetFrameSamples: (split: string | undefined, dataset: string | undefined, offset: number, sequenceId?: string, limit?: number) =>
    ["api-v1", "dataset", "frame-samples", split, dataset, offset, sequenceId, limit] as const,
  realDatasetEvaluation: (split?: string, imageId?: string) =>
    ["api-v1", "dataset", "evaluation", split, imageId] as const,
  annotations: (split?: string, imageId?: string) =>
    ["api-v1", "dataset", "annotations", split, imageId] as const,
  annotationHistory: (split?: string, imageId?: string) =>
    ["api-v1", "dataset", "annotations", split, imageId, "history"] as const,
  applicationUsers: ["api-v1", "auth", "users"] as const,
  adminProjects: ["api-v1", "control", "projects"] as const,
  adminBatches: (projectId?: string) => ["api-v1", "control", "batches", projectId] as const,
  teamHealth: (projectId?: string) => ["api-v1", "control", "team-health", projectId] as const,
};

export function useApplicationUsersQuery(enabled = true) {
  return useQuery({
    queryKey: apiQueryKeys.applicationUsers,
    queryFn: ({ signal }) => labelGuardianApiV1.listApplicationUsers(signal),
    enabled,
  });
}

export function useUpdateApplicationUserRoleMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ userId, role }: { userId: string; role: "annotator" | "reviewer" | "admin" }) =>
      labelGuardianApiV1.updateApplicationUserRole(userId, role),
    onSuccess: async () => queryClient.invalidateQueries({ queryKey: apiQueryKeys.applicationUsers }),
  });
}

export function useUpdateApplicationUserStatusMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ userId, disabled }: { userId: string; disabled: boolean }) =>
      labelGuardianApiV1.updateApplicationUserStatus(userId, disabled),
    onSuccess: async () => queryClient.invalidateQueries({ queryKey: apiQueryKeys.applicationUsers }),
  });
}

export function useInviteApplicationUserMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: { email: string; displayName: string; role: "annotator" | "reviewer" | "admin" }) =>
      labelGuardianApiV1.inviteApplicationUser(payload),
    onSuccess: async () => queryClient.invalidateQueries({ queryKey: apiQueryKeys.applicationUsers }),
  });
}

export function useAdminProjectsQuery(enabled = true) {
  return useQuery({
    queryKey: apiQueryKeys.adminProjects,
    queryFn: ({ signal }) => labelGuardianApiV1.listAdminProjects(signal),
    enabled,
  });
}

export function useAdminBatchesQuery(projectId?: string, enabled = true) {
  return useQuery({
    queryKey: apiQueryKeys.adminBatches(projectId),
    queryFn: ({ signal }) => labelGuardianApiV1.listAdminBatches(projectId, signal),
    enabled,
  });
}

export function useTeamHealthQuery(projectId?: string, enabled = true) {
  return useQuery({
    queryKey: apiQueryKeys.teamHealth(projectId),
    queryFn: ({ signal }) => labelGuardianApiV1.getTeamHealth(projectId, signal),
    enabled,
  });
}

export function useQaCasesQuery(
  filters: { split?: string; datasetId?: string; sourceImageId?: string } = {},
  enabled = true,
) {
  return useQuery({
    queryKey: filters.sourceImageId
      ? apiQueryKeys.qaCasesForImage(filters.split, filters.sourceImageId)
      : [...apiQueryKeys.qaCases, filters.datasetId, filters.split],
    queryFn: ({ signal }) => labelGuardianApiV1.listQaCases(signal, filters),
    enabled,
  });
}

export function useRealDatasetImagesQuery(split: string | undefined, offset: number, dataset?: string) {
  return useQuery({
    queryKey: apiQueryKeys.realDatasetImages(split, dataset, offset),
    queryFn: ({ signal }) => labelGuardianApiV1.listRealDatasetImages(split, offset, signal, dataset),
  });
}

export function useRealDatasetFrameSamplesQuery(split: string | undefined, offset: number, dataset?: string, sequenceId?: string, limit?: number) {
  return useQuery({
    queryKey: apiQueryKeys.realDatasetFrameSamples(split, dataset, offset, sequenceId, limit),
    queryFn: ({ signal }) => labelGuardianApiV1.listRealDatasetFrameSamples(split, offset, signal, dataset, sequenceId, limit),
  });
}

export function useRealDatasetImageEvaluationQuery(split?: string, imageId?: string, enabled = true) {
  return useQuery({
    queryKey: apiQueryKeys.realDatasetEvaluation(split, imageId),
    queryFn: ({ signal }) =>
      labelGuardianApiV1.getRealDatasetImageEvaluation(split!, imageId!, signal),
    enabled: enabled && Boolean(split && imageId),
    staleTime: 15_000,
  });
}

export function useEvaluateRealDatasetImageMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      split,
      imageId,
      force = false,
      persist = true,
    }: {
      split: string;
      imageId: string;
      force?: boolean;
      persist?: boolean;
    }) =>
      labelGuardianApiV1.evaluateRealDatasetImage(
        split,
        imageId,
        force,
        persist,
      ),
    onSuccess: async (data, variables) => {
      queryClient.setQueryData(
        apiQueryKeys.realDatasetEvaluation(variables.split, variables.imageId),
        data,
      );
      await queryClient.invalidateQueries({ queryKey: apiQueryKeys.qaCases });
    },
  });
}

export function useEvaluateRealDatasetBatchMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      split,
      imageIds,
      force = false,
      persist = true,
    }: {
      split: string;
      imageIds: string[];
      force?: boolean;
      persist?: boolean;
    }) =>
      labelGuardianApiV1.evaluateRealDatasetImagesBatch(
        split,
        imageIds,
        force,
        persist,
      ),
    onSuccess: async (data, variables) => {
      data.results.forEach((result) => {
        if (!result.evaluation) return;
        queryClient.setQueryData(
          apiQueryKeys.realDatasetEvaluation(variables.split, result.imageId),
          result.evaluation,
        );
      });
      await queryClient.invalidateQueries({ queryKey: apiQueryKeys.qaCases });
    },
  });
}

export function useQaCaseStatusMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      caseId,
      status,
      actorId,
      reason,
    }: {
      caseId: string;
      status: "in_review" | "confirmed" | "rejected" | "skipped";
      actorId?: string;
      reason?: string;
    }) =>
      labelGuardianApiV1.updateQaCaseStatus(caseId, status, actorId, reason),
    onSuccess: async () =>
      queryClient.invalidateQueries({ queryKey: apiQueryKeys.qaCases }),
  });
}

export function useImageAnnotationsQuery(split?: string, imageId?: string) {
  return useQuery({
    queryKey: apiQueryKeys.annotations(split, imageId),
    queryFn: ({ signal }) =>
      labelGuardianApiV1.getImageAnnotations(split!, imageId!, signal),
    enabled: Boolean(split && imageId),
  });
}

export function useAnnotationHistoryQuery(split?: string, imageId?: string) {
  return useQuery({
    queryKey: apiQueryKeys.annotationHistory(split, imageId),
    queryFn: ({ signal }) =>
      labelGuardianApiV1.getImageAnnotationHistory(split!, imageId!, signal),
    enabled: Boolean(split && imageId),
  });
}

export function useSaveAnnotationsMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      split,
      imageId,
      expectedRevision,
      labels,
      actorId,
      changeNote,
    }: {
      split: string;
      imageId: string;
      expectedRevision: number;
      labels: RealDatasetLabelDto[];
      actorId?: string;
      changeNote?: string;
    }) =>
      labelGuardianApiV1.saveImageAnnotations(split, imageId, {
        expectedRevision,
        labels,
        actorId,
        changeNote,
      }),
    onSuccess: async (_data, variables) => {
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: apiQueryKeys.annotations(
            variables.split,
            variables.imageId,
          ),
        }),
        queryClient.invalidateQueries({
          queryKey: apiQueryKeys.annotationHistory(
            variables.split,
            variables.imageId,
          ),
        }),
        queryClient.invalidateQueries({ queryKey: ["api-v1", "dataset"] }),
        queryClient.invalidateQueries({ queryKey: apiQueryKeys.qaCases }),
      ]);
    },
  });
}

export function useRestoreAnnotationsMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      split,
      imageId,
      expectedRevision,
      targetRevision,
      actorId,
    }: {
      split: string;
      imageId: string;
      expectedRevision: number;
      targetRevision: number;
      actorId?: string;
    }) =>
      labelGuardianApiV1.restoreImageAnnotations(split, imageId, {
        expectedRevision,
        targetRevision,
        actorId,
      }),
    onSuccess: async (_data, variables) => {
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: apiQueryKeys.annotations(
            variables.split,
            variables.imageId,
          ),
        }),
        queryClient.invalidateQueries({
          queryKey: apiQueryKeys.annotationHistory(
            variables.split,
            variables.imageId,
          ),
        }),
        queryClient.invalidateQueries({ queryKey: ["api-v1", "dataset"] }),
        queryClient.invalidateQueries({ queryKey: apiQueryKeys.qaCases }),
      ]);
    },
  });
}

export function usePipelineRunsQuery(enabled = true) {
  return useQuery({
    queryKey: ["api-v1", "ingestion", "runs"],
    queryFn: ({ signal }) => labelGuardianApiV1.listPipelineRuns(signal),
    enabled,
  });
}
