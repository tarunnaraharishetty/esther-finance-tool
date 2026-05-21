import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide transition-colors",
  {
    variants: {
      variant: {
        default: "border-border/60 bg-secondary text-secondary-foreground",
        bull: "border-bull/40 bg-bull/10 text-bull",
        bear: "border-bear/40 bg-bear/10 text-bear",
        warn: "border-warn/40 bg-warn/10 text-warn",
        accent: "border-accent/40 bg-accent/10 text-accent",
        outline: "border-border/60 bg-transparent text-muted-foreground",
      },
    },
    defaultVariants: { variant: "default" },
  },
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

export function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <span className={cn(badgeVariants({ variant }), className)} {...props} />
  );
}

export { badgeVariants };
