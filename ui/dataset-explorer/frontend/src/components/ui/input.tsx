import * as React from "react";
import { cn } from "@/lib/utils";

const sharedClasses =
  "flex w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background transition-shadow placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50";

const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className, type, ...props }, ref) => (
    <input type={type} className={cn(sharedClasses, className)} ref={ref} {...props} />
  ),
);
Input.displayName = "Input";

const NativeSelect = React.forwardRef<
  HTMLSelectElement,
  React.SelectHTMLAttributes<HTMLSelectElement>
>(({ className, ...props }, ref) => (
  <select className={cn(sharedClasses, className)} ref={ref} {...props} />
));
NativeSelect.displayName = "NativeSelect";

export { Input, NativeSelect };
