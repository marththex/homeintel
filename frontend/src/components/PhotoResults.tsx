import { useMediaQuery } from "../hooks/useMediaQuery";
import { PhotoCarousel, type CarouselPhoto } from "./PhotoCarousel";
import { PhotoGrid } from "./PhotoGrid";

interface Props {
  photos: CarouselPhoto[];
}

// The single branching point: grid on ≥768px, swipeable carousel on mobile.
export function PhotoResults({ photos }: Props) {
  const isWide = useMediaQuery("(min-width: 768px)");
  return isWide ? <PhotoGrid photos={photos} /> : <PhotoCarousel photos={photos} />;
}
