/**
 * Decorative backdrop: a convex-probe ultrasound sector with depth arcs and a
 * slow outward echo. Purely ornamental, hidden from assistive technology.
 */
const ORIGIN = { x: 1080, y: -60 };
const SPREAD = 35; // degrees either side of straight down
const OUTER = 900;

function polar(radius, degrees) {
  const radians = (degrees * Math.PI) / 180;
  return { x: ORIGIN.x + radius * Math.cos(radians), y: ORIGIN.y + radius * Math.sin(radians) };
}

function arc(radius) {
  const right = polar(radius, 90 - SPREAD);
  const left = polar(radius, 90 + SPREAD);
  return `M ${right.x} ${right.y} A ${radius} ${radius} 0 0 1 ${left.x} ${left.y}`;
}

const RADII = Array.from({ length: 10 }, (_, index) => 180 + index * 80);
const RAYS = [-SPREAD, -SPREAD / 2, 0, SPREAD / 2, SPREAD];

export default function HeroBackdrop() {
  const right = polar(OUTER, 90 - SPREAD);
  const left = polar(OUTER, 90 + SPREAD);
  return (
    <div
      aria-hidden="true"
      className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-[780px] overflow-hidden opacity-50 sm:opacity-100 [mask-image:linear-gradient(to_bottom,black_45%,transparent)]"
    >
      <svg viewBox="0 0 1440 780" preserveAspectRatio="xMaxYMin slice" className="h-full w-full">
        <defs>
          <radialGradient id="sector-fill" gradientUnits="userSpaceOnUse" cx={ORIGIN.x} cy={ORIGIN.y} r={OUTER}>
            <stop offset="0" stopColor="#2dd4bf" stopOpacity="0.16" />
            <stop offset="0.55" stopColor="#2dd4bf" stopOpacity="0.05" />
            <stop offset="1" stopColor="#2dd4bf" stopOpacity="0" />
          </radialGradient>
        </defs>
        <path d={`M ${ORIGIN.x} ${ORIGIN.y} L ${right.x} ${right.y} A ${OUTER} ${OUTER} 0 0 1 ${left.x} ${left.y} Z`} fill="url(#sector-fill)" />
        <g fill="none" stroke="#5eead4" strokeWidth="1" vectorEffect="non-scaling-stroke">
          {RADII.map((radius, index) => (
            <path key={radius} d={arc(radius)} strokeOpacity={0.16 - index * 0.012} strokeDasharray={index % 2 ? '2 7' : undefined} />
          ))}
          {RAYS.map((offset) => {
            const start = polar(120, 90 + offset);
            const end = polar(OUTER, 90 + offset);
            return <line key={offset} x1={start.x} y1={start.y} x2={end.x} y2={end.y} strokeOpacity="0.07" strokeDasharray="1 9" />;
          })}
        </g>
        <path
          d={arc(OUTER)}
          fill="none"
          stroke="#5eead4"
          strokeWidth="1.5"
          vectorEffect="non-scaling-stroke"
          className="sonar-echo"
          style={{ transformOrigin: `${ORIGIN.x}px ${ORIGIN.y}px` }}
        />
      </svg>
    </div>
  );
}
