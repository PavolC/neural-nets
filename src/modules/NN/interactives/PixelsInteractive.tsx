import { useEffect, useRef, useState } from "react";
import { fetchMnistTest, type MnistTestSubset } from "../../../runtime/assets";
import { drawMnistDigit } from "./utils";
import { useInViewOnce } from "../../../components/useInViewOnce";

// Chapter 2 interactive: an image is just numbers. One real MNIST test
// digit, blown up; pointing at any pixel shows its brightness value and
// its position in the unrolled (784, 1) input column.

const DIGIT_INDEX = 0; // the first test digit (a 7)
const N = 28;
const SIZE = 280;

export function PixelsInteractive() {
  const [mnist, setMnist] = useState<MnistTestSubset | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [cell, setCell] = useState<{ r: number; c: number } | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hostRef = useRef<HTMLDivElement>(null);
  const inView = useInViewOnce(hostRef);

  useEffect(() => {
    if (!inView) return;
    fetchMnistTest().then(setMnist).catch((err) => setLoadError(String(err)));
  }, [inView]);

  useEffect(() => {
    if (mnist && canvasRef.current) drawMnistDigit(canvasRef.current, mnist.images, DIGIT_INDEX);
  }, [mnist]);

  // Pointer, not mouse: the same handler then covers a finger and a stylus.
  // With onMouseMove this whole panel did nothing at all on a phone, and its
  // status line read "Point anywhere on the image." forever.
  const onMove = (e: React.PointerEvent) => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const c = Math.min(N - 1, Math.max(0, Math.floor(((e.clientX - rect.left) / rect.width) * N)));
    const r = Math.min(N - 1, Math.max(0, Math.floor(((e.clientY - rect.top) / rect.height) * N)));
    setCell({ r, c });
  };

  if (loadError)
    return (
      <div className="interactive" ref={hostRef}>
        <p className="demo-status demo-status-error">
          Could not load the digit: {loadError}
        </p>
      </div>
    );
  if (!mnist)
    return (
      <div className="interactive" ref={hostRef}>
        <p className="demo-status">Loading a digit...</p>
      </div>
    );

  const value = cell ? mnist.images[DIGIT_INDEX * 784 + cell.r * N + cell.c] / 255 : 0;
  const label = mnist.labels[DIGIT_INDEX];

  return (
    <div className="interactive" ref={hostRef}>
      <div className="pixels-row">
        <div
          className="pixels-stack"
          onPointerMove={onMove}
          onPointerDown={onMove}
          // A finger lifting is not the reader losing interest: keep the last
          // pixel on screen so the numbers can be read after the tap.
          onPointerLeave={(e) => {
            if (e.pointerType === "mouse") setCell(null);
          }}
        >
          <canvas
            ref={canvasRef}
            width={N}
            height={N}
            style={{ width: SIZE, height: SIZE }}
            role="img"
            aria-label={`A handwritten ${label} from the test set, drawn as 28 rows of 28 brightness values.`}
          />
          <svg viewBox={`0 0 ${N} ${N}`} preserveAspectRatio="none" aria-hidden="true">
            {cell && (
              <rect x={cell.c} y={cell.r} width={1} height={1} className="pixel-highlight" />
            )}
          </svg>
        </div>
        <div className="pixels-info">
          <p className="interactive-status status-fixed">
            {cell
              ? `The pixel at row ${cell.r}, column ${cell.c} holds the number ${value.toFixed(2)}. ` +
                `Unrolled, it is entry ${cell.r * N + cell.c} of the 784-tall input column ` +
                `(row ${cell.r} times 28, plus column ${cell.c}).`
              : "Point at the image, or tap it, to read a pixel."}
          </p>
          <p className="pixels-note">
            This is a real test digit (its label: {label}). To you it is a {label}; to the
            network it is 28 rows of 28 brightness numbers, 0 for blank paper, 1 for full
            ink, read row by row into one long column of 784.
          </p>
        </div>
      </div>
    </div>
  );
}
