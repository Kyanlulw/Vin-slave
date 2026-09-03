import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import type { RealDatasetEvaluationDto } from "../src/api/types.ts";
import {
  apiBoxIntersectsImage,
  boxIntersectsImage,
  reportForSelectedImage,
} from "../src/utils/realDataset.ts";

const readSource = (relativePath: string) =>
  readFileSync(new URL(relativePath, import.meta.url), "utf8");

test("real data view does not read a report before an evaluation exists", () => {
  assert.equal(reportForSelectedImage(undefined, undefined), undefined);
  assert.equal(reportForSelectedImage(undefined, "000000"), undefined);
});

test("real data view only shows the report for the selected image", () => {
  const evaluation = {
    image: { id: "000000" },
    report: { status: "pass", summary: "ok", metrics: {}, issues: [] },
  } as RealDatasetEvaluationDto;

  assert.equal(reportForSelectedImage(evaluation, "000001"), undefined);
  assert.equal(reportForSelectedImage(evaluation, "000000"), evaluation.report);
});

test("display filtering hides only boxes that do not intersect the image", () => {
  assert.equal(
    boxIntersectsImage({ x: 10, y: 10, width: 20, height: 20 }, 100, 100),
    true,
  );
  assert.equal(
    boxIntersectsImage({ x: -10, y: 10, width: 20, height: 20 }, 100, 100),
    true,
  );
  assert.equal(
    boxIntersectsImage({ x: -30, y: 10, width: 20, height: 20 }, 100, 100),
    false,
  );
  assert.equal(
    apiBoxIntersectsImage({ x1: 110, y1: 10, x2: 130, y2: 30 }, 100, 100),
    false,
  );
});

test("real data dataset selector keeps both official datasets available", () => {
  const realDataSource = readSource("../src/views/RealDataQAView.tsx");

  assert.match(realDataSource, /REAL_DATA_DATASET_OPTIONS = \["nuscenes", "kitti"\]/);
  assert.match(realDataSource, /new Set<string>\(REAL_DATA_DATASET_OPTIONS\)/);
  assert.match(realDataSource, /availableDatasets\.forEach/);
});
