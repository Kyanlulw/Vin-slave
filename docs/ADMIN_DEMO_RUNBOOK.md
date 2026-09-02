# Admin Control Plane — Demo Runbook

## Preflight

Confirm the backend production environment contains these backend-only values:

- `AUTH_ENABLED=true`
- `SUPABASE_URL` and `SUPABASE_JWT_AUDIENCE=authenticated`
- `SUPABASE_SERVICE_ROLE_KEY` (required for invite/disable/enable)
- `DATABASE_URL` pointing at the production Supabase project
- `LABEL_GUARDIAN_STORAGE_BACKEND=gcs`
- `LABEL_GUARDIAN_GCS_BUCKET` and `LABEL_GUARDIAN_GCS_PROJECT`
- `CORS_ORIGINS` includes the production Vercel origin

The browser may contain only `VITE_*` values. Never put a service-role key,
database URL, GCS credential, or JWT secret in `frontend/.env*`.

## Demo flow

1. Sign in with the bootstrap admin account. Open `/admin`.
2. In **Team**, invite one annotator and one reviewer. Set roles and verify
   both accounts are active.
3. In **Projects & Data**, create a customer project, create a submission,
   add an archive, and upload it. For a bucket import, use only a prefix in
   the configured bucket allowlist.
4. Complete every asset upload, then start ingestion. Wait for the submission
   to reach `ready` and inspect the validation/ingestion result.
5. Create a batch from the published frame inventory. Choose the reviewer and
   annotator pool; confirm the task preview before creating it.
6. As annotator, open only assigned tasks: `assigned → in_progress → submitted`.
7. As reviewer, open the submitted task and choose either:
   - `request changes` with a required reason, then have the annotator
     `resubmitted`; or
   - `approve` directly.
8. As admin, use Team Health drill-down to verify stage counts, WIP and
   rework. Freeze a release only after every task in scope is approved.
9. Export the YOLO 2D package, create a short-lived signed download URL, and
   verify the manifest/checksum and audit history.

## Security checks

- An annotator cannot list or stream another user's task/media (the API returns
  `404`).
- A reviewer cannot review a task outside the assigned batch (the API returns
  `404`).
- Invalid stage transitions return `409`; requesting changes without a reason
  returns `422`.
- The final active admin cannot be demoted or disabled.
- The bucket remains private and the browser has no privileged credential.

## Useful smoke commands

```bash
curl -fsS https://api.labelguardian.space/health
curl -fsS https://api.labelguardian.space/ready
curl -fsS https://api.labelguardian.space/openapi.json | jq '.paths | with_entries(select(.key | startswith("/api/v1/control"))) | keys'
```

