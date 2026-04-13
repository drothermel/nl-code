import { cva, type VariantProps } from "class-variance-authority";
import * as React from "react";
import { cn } from "@/lib/utils";

const codeBlockVariants = cva("overflow-x-auto rounded-md border p-4 text-xs leading-6", {
  variants: {
    variant: {
      default: "bg-card text-card-foreground",
      error: "border-destructive/30 bg-card text-destructive",
    },
  },
  defaultVariants: { variant: "default" },
});

interface CodeBlockProps
  extends React.HTMLAttributes<HTMLPreElement>,
    VariantProps<typeof codeBlockVariants> {}

const CodeBlock = React.forwardRef<HTMLPreElement, CodeBlockProps>(
  ({ className, variant, ...props }, ref) => (
    <pre ref={ref} className={cn(codeBlockVariants({ variant }), className)} {...props} />
  ),
);
CodeBlock.displayName = "CodeBlock";

export { CodeBlock, codeBlockVariants };
