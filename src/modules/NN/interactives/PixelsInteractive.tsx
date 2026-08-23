import { useEffect, useRef, useState } from "react";
import { fetchMnistTest, type MnistTestSubset } from "../../../runtime/assets";
import { drawMnistDigit } from "./utils";

// Module 2 interactive: an image is just numbers. One real MNIST test
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

  useEffect(() => {
    fetchMnistTest().then(setMnist).catch((err) => setLoadError(String(err)));
  }, []);

  useEffect(() => {
    if (mnist && canvasRef.current) drawMnistDigit(canvasRef.current, mnist.images, DIGIT_INDEX);
  }, [mnist]);

  if (loadError) return <p className="demo-status demo-status-error">Could not load the digit: {loadError}</p>;
  if (!mnist) return <p className="demo-status">Loading a digit...</p>;

  const onMove = (e: React.MouseEvent) => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const c = Math.min(N - 1, Math.max(0, Math.floor(((e.clientX - rect.left) / rect.width) * N)));
    const r = Math.min(N - 1, Math.max(0, Math.floor(((e.clientY - rect.top) / rect.height) * N)));
    setCell({ r, c });
  };

  const value = cell ? mnist.images[DIGIT_INDEX * 784 + cell.r * N + cell.c] / 255 : 0;
  const label = mnist.labels[DIGIT_INDEX];

  return (
    <div className="interactive">
      <div className="pixels-row">
        <div
          className="pixels-stack"
          onMouseMove={onMove}
          onMouseLeave={() => setCell(null)}
        >
          <canvas ref={canvasRef} width={N} height={N} style={{ width: SIZE, height: SIZE }} />
          <svg viewBox={`0 0 ${N} ${N}`} preserveAspectRatio="none">
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
              : "Point anywhere on the image."}
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
