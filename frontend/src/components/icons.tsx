// Lightweight inline SVG icons — self-contained, crisp at any size.
// Stroke uses currentColor so they inherit text color.

interface IconProps {
  size?: number;
  className?: string;
}

const base = (size: number, className?: string) => ({
  width: size,
  height: size,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  className,
});

export const PlusIcon = ({ size = 22, className }: IconProps) => (
  <svg {...base(size, className)}><path d="M12 5v14M5 12h14" /></svg>
);

export const SendIcon = ({ size = 20, className }: IconProps) => (
  <svg {...base(size, className)}><path d="M12 19V5M5 12l7-7 7 7" /></svg>
);

export const ComposeIcon = ({ size = 20, className }: IconProps) => (
  <svg {...base(size, className)}>
    <path d="M12 20h9" />
    <path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z" />
  </svg>
);

export const CameraIcon = ({ size = 22, className }: IconProps) => (
  <svg {...base(size, className)}>
    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
    <circle cx="12" cy="13" r="4" />
  </svg>
);

export const ImageIcon = ({ size = 22, className }: IconProps) => (
  <svg {...base(size, className)}>
    <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
    <circle cx="8.5" cy="8.5" r="1.5" />
    <path d="m21 15-5-5L5 21" />
  </svg>
);

export const CloseIcon = ({ size = 18, className }: IconProps) => (
  <svg {...base(size, className)}><path d="M18 6 6 18M6 6l12 12" /></svg>
);

export const FilterIcon = ({ size = 20, className }: IconProps) => (
  <svg {...base(size, className)}><path d="M22 3H2l8 9.46V19l4 2v-8.54L22 3z" /></svg>
);
