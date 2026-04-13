import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneLight } from "react-syntax-highlighter/dist/esm/styles/prism";
import { cn } from "@/lib/utils";

export default function PythonCodeBlock({ code, className }: { code: string; className?: string }) {
  return (
    <SyntaxHighlighter
      language="python"
      style={oneLight}
      wrapLongLines
      className={cn("overflow-x-auto rounded-md border text-xs text-card-foreground", className)}
      customStyle={{
        margin: 0,
        borderRadius: "0.375rem",
        padding: "1rem",
        background: "hsl(var(--card))",
        fontSize: "0.75rem",
        lineHeight: "1.5rem",
      }}
    >
      {code}
    </SyntaxHighlighter>
  );
}
