import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import Layout from "@/components/layout/Layout";
import ComparePage from "@/pages/ComparePage";
import HomeRedirect from "@/pages/HomeRedirect";
import OverviewPage from "@/pages/OverviewPage";
import RawDetailPage from "@/pages/RawDetailPage";
import TaskBrowserPage from "@/pages/TaskBrowserPage";
import TaskDetailPage from "@/pages/TaskDetailPage";

export default function App() {
  return (
    <BrowserRouter>
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
    </BrowserRouter>
  );
}
