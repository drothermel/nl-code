import type { ReactNode } from "react";

export function PageLoading({ label, skeleton }: { label: string; skeleton?: ReactNode }) {
  if (skeleton) return <>{skeleton}</>;
  return <div className="p-8 text-sm text-muted-foreground">Loading {label}...</div>;
}

export function PageError({ label, error }: { label: string; error?: Error | null }) {
  return (
    <div className="p-8 text-sm text-destructive">
      Failed to load {label}: {error?.message ?? "Unknown error"}
    </div>
  );
}
