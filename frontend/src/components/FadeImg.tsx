import { useState } from "react";

type Props = React.ImgHTMLAttributes<HTMLImageElement>;

// Shimmering placeholder that fades the image in on load. The box itself is
// reserved by the parent's fixed aspect ratio, so loading causes zero shift.
export function FadeImg({ className = "", ...rest }: Props) {
  const [loaded, setLoaded] = useState(false);
  return (
    <span className={`img-frame ${loaded ? "" : "skel"}`}>
      <img
        {...rest}
        className={`${className} fade-img ${loaded ? "loaded" : ""}`}
        onLoad={() => setLoaded(true)}
      />
    </span>
  );
}
