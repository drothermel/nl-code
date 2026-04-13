import { lazy, Suspense } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import Layout from "@/components/layout/Layout";
import HomeRedirect from "@/pages/HomeRedirect";

const ComparePage = lazy(() => import("@/pages/ComparePage"));
const OverviewPage = lazy(() => import("@/pages/OverviewPage"));
const RawDetailPage = lazy(() => import("@/pages/RawDetailPage"));
const TaskBrowserPage = lazy(() => import("@/pages/TaskBrowserPage"));
const TaskDetailPage = lazy(() => import("@/pages/TaskDetailPage"));

function PageFallback() {
  return <div className="p-8 text-sm text-muted-foreground">Loading…</div>;
}

export default function App() {
  return (
    <BrowserRouter>
      <Suspense fallback={<PageFallback />}>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<HomeRedirect />} />
            <Route path="/compare" element={<ComparePage />} />
            <Route path="/datasets/:datasetKey/overview" element={<OverviewPage />} />
            <Route path="/datasets/:datasetKey/tasks" element={<TaskBrowserPage />} />
            <Route path="/datasets/:datasetKey/tasks/*" element={<TaskDetailPage />} />
            <Route path="/datasets/:datasetKey/raw/*" element={<RawDetailPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}
