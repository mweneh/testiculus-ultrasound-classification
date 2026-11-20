import { BookOpen, Database, ExternalLink, FileText, HeartPulse, ScanSearch, ShieldAlert, Split, Video, Waves } from 'lucide-react';
import { comparison } from '../lib/records.js';
import { grouped } from '../lib/format.js';
import { Reveal, SectionHeading } from './ui.jsx';

export const REPOSITORY_URL = 'https://github.com/mweneh/testiculus-ultrasound-classification';
export const REPORT_URL = `${REPOSITORY_URL}/blob/main/report/capstone_report.pdf`;
export const DATASET_URL = 'https://ditto.ing.unimore.it/testiculus/';
export const LICENCE_URL = 'https://creativecommons.org/licenses/by-nc-sa/4.0/';
export const ICIAP_DOI_URL = 'https://doi.org/10.1007/978-3-032-10192-1_3';

function ClassBar({ negative, positive, label }) {
  const total = negative + positive;
  return (
    <div>
      <div className="flex items-baseline justify-between text-[12px]">
        <span className="text-ink-2">{label}</span>
        <span className="tabular font-mono text-ink-1">{grouped(total)}</span>
      </div>
      <div
        className="mt-1.5 flex h-2 gap-[2px] overflow-hidden rounded-full"
        role="img"
        aria-label={`${label}: ${negative} homogeneous, ${positive} inhomogeneous`}
      >
        <span className="h-full rounded-l-full bg-ink-3/70" style={{ width: `${(negative / total) * 100}%` }} />
        <span className="h-full rounded-r-full bg-accent/80" style={{ width: `${(positive / total) * 100}%` }} />
      </div>
      <div className="mt-1 flex justify-between font-mono text-[10.5px] text-ink-3">
        <span>{negative} homogeneous</span>
        <span>{positive} inhomogeneous</span>
      </div>
    </div>
  );
}

/** Deterministic pseudo-random jitter in [0, 1) so the schematic renders identically every time. */
function jitter(index) {
  const value = Math.sin(index * 12.9898) * 43758.5453;
  return value - Math.floor(value);
}

const BLOBS = [
  { x: 4, y: 5, r: 3.2, delta: -0.24 },
  { x: 11, y: 4, r: 2.6, delta: 0.3 },
  { x: 9, y: 11, r: 3.6, delta: -0.2 },
  { x: 3, y: 12, r: 2.2, delta: 0.26 },
];

/** Abstract schematic of an even versus a patchy texture; not an ultrasound image. */
function TextureSwatch({ patchy, label }) {
  const size = 14;
  const cells = Array.from({ length: size * size }, (_, index) => {
    const x = index % size;
    const y = Math.floor(index / size);
    let level = 0.42 + (jitter(index) - 0.5) * 0.12;
    if (patchy) {
      for (const blob of BLOBS) {
        const distance = Math.hypot(x - blob.x, y - blob.y);
        level += blob.delta * Math.exp(-(distance * distance) / (2 * blob.r * blob.r)) * 1.6;
      }
      level += (jitter(index + 999) - 0.5) * 0.18;
    }
    return { x, y, level: Math.min(0.95, Math.max(0.04, level)) };
  });
  return (
    <figure className="flex flex-col items-center gap-1.5">
      <svg viewBox={`0 0 ${size} ${size}`} className="h-16 w-16 rounded-lg ring-1 ring-lab-line" aria-hidden="true">
        <rect width={size} height={size} fill="#04080f" />
        {cells.map((cell) => (
          <rect key={`${cell.x}-${cell.y}`} x={cell.x} y={cell.y} width="1" height="1" fill="#e8eef6" fillOpacity={cell.level} />
        ))}
      </svg>
      <figcaption className="text-[11.5px] text-ink-2">{label}</figcaption>
    </figure>
  );
}

function LabelPoint({ icon: Icon, title, children }) {
  return (
    <div>
      <h4 className="flex items-center gap-2 text-[13.5px] font-semibold text-ink-1">
        <Icon aria-hidden="true" className="h-3.5 w-3.5 text-accent" />
        {title}
      </h4>
      <div className="mt-2 space-y-2">{children}</div>
    </div>
  );
}

function Card({ icon: Icon, title, children, className = '', delay = 0 }) {
  return (
    <Reveal as="article" delay={delay} className={`panel p-5 ${className}`}>
      <h3 className="flex items-center gap-2 text-[15px] font-semibold text-ink-1">
        <Icon aria-hidden="true" className="h-4 w-4 text-accent" />
        {title}
      </h3>
      <div className="mt-3 text-[13.5px] leading-relaxed text-ink-2">{children}</div>
    </Reveal>
  );
}

function ExternalAnchor({ href, children }) {
  return (
    <a href={href} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 font-medium text-accent-soft underline-offset-2 hover:underline">
      {children}
      <ExternalLink aria-hidden="true" className="h-3 w-3" />
      <span className="sr-only">(opens in a new tab)</span>
    </a>
  );
}

export default function AboutSection() {
  const { dataset, splits } = comparison;
  return (
    <section id="about" aria-labelledby="about-title" className="scroll-mt-28">
      <SectionHeading id="about-title" eyebrow="About the research" title="What this prototype is, and what it is not">
        <p>
          A comparison of image-level classifiers for testicular ultrasound appearance. The target is the released homogeneous / inhomogeneous image label, not a patient
          diagnosis, infertility or malignancy.
        </p>
      </SectionHeading>

      <div className="mt-8 grid gap-4 lg:grid-cols-3">
        <Card icon={ScanSearch} title="What the labels mean" className="lg:col-span-3">
          <div className="grid gap-6 md:grid-cols-3">
            <LabelPoint icon={Waves} title="Tissue texture">
              <p>
                <strong className="font-semibold text-ink-1">Homogeneous</strong>: an even, uniform echotexture, with fine echoes
                spread consistently across the testicular tissue.{' '}
                <strong className="font-semibold text-ink-1">Inhomogeneous</strong>: an uneven texture, with patchy or mottled
                regions darker or brighter than their surroundings.
              </p>
              <div className="flex gap-5 pt-1">
                <TextureSwatch label="Even" />
                <TextureSwatch patchy label="Patchy" />
              </div>
              <p className="text-[12px] text-ink-3">
                Schematic only, not ultrasound images. The dataset authors publish no visual criteria; the labels are
                clinicians&apos; judgements.
              </p>
            </LabelPoint>
            <LabelPoint icon={HeartPulse} title="Why it matters">
              <p>
                The dataset authors describe parenchymal inhomogeneity as &ldquo;an emerging biomarker for male
                infertility&rdquo; (Morelli et al., ICIAP 2025), citing Spaggiari et al., <em>Asian Journal of Andrology</em>,
                2020.
              </p>
              <p>
                It is a finding that informs a fertility assessment, not a diagnosis of infertility or any other condition.
                This tool predicts the appearance label only.
              </p>
            </LabelPoint>
            <LabelPoint icon={Video} title="Judged on video, saved as stills">
              <p>
                Clinicians assess homogeneity during a live video examination, but only still screenshots are kept, and a
                single frame may not show what they saw.
              </p>
              <p>
                When the authors had clinicians re-review 72 suspicious images from the stills alone, the new assessments did
                not agree with the original labels. Some labels are noisy, so some apparent model errors may be label errors.
              </p>
            </LabelPoint>
          </div>
        </Card>

        <Card icon={Database} title="Dataset: TesticulUS-Real">
          <p>
            {grouped(dataset.total)} labelled grayscale images: {grouped(dataset.negative)} homogeneous (class 0) and{' '}
            {grouped(dataset.positive)} inhomogeneous (class 1, positive). About {(dataset.negative / dataset.positive).toFixed(1)} : 1
            imbalance. 200 unlabelled images and the segmentation masks were not used as classification inputs.
          </p>
          <div className="mt-4">
            <ClassBar label="All labelled images" negative={dataset.negative} positive={dataset.positive} />
          </div>
        </Card>

        <Card icon={Split} title="Image-level split" delay={90}>
          <p>Fixed seed 42, stratified ~70/15/15. Thresholds were selected on validation only; the test partition was not used for tuning.</p>
          <div className="mt-4 space-y-3">
            <ClassBar label="Training" negative={splits.train.negative} positive={splits.train.positive} />
            <ClassBar label="Validation" negative={splits.validation.negative} positive={splits.validation.positive} />
            <ClassBar label="Test" negative={splits.test.negative} positive={splits.test.positive} />
          </div>
        </Card>

        <Card icon={ShieldAlert} title="Limitations" delay={180}>
          <ul className="list-disc space-y-1.5 pl-4 marker:text-ink-3">
            <li>Patient, examination and centre identifiers were unavailable: results are not patient-independent or cross-centre.</li>
            <li>Only {splits.test.positive} inhomogeneous test images; one training seed; no confidence intervals.</li>
            <li>Scores are uncalibrated; thresholds optimize F1, not a clinical cost.</li>
            <li>Grad-CAM was verified technically; its anatomical validity was not assessed.</li>
            <li>Not clinically validated. Not for diagnosis.</li>
          </ul>
        </Card>

        <Card icon={BookOpen} title="Dataset attribution and citations" className="lg:col-span-2">
          <p>
            Dataset credit belongs to Morelli and colleagues and the TesticulUS contributors (
            <ExternalAnchor href={DATASET_URL}>official dataset site</ExternalAnchor>). Images and adapted visualizations, including
            Grad-CAM overlays, are licensed under <ExternalAnchor href={LICENCE_URL}>CC BY-NC-SA 4.0</ExternalAnchor>: noncommercial
            use, attribution and ShareAlike apply.
          </p>
          <ol className="mt-4 space-y-3 border-l border-lab-line pl-4 text-[13px]">
            <li>
              Morelli N., Marchesini K., Lumetti L., Santi D., Grana C., Bolelli F. <em>Enhancing Testicular Ultrasound Image
              Classification Through Synthetic Data and Pretraining Strategies.</em> Image Analysis and Processing – ICIAP 2025,
              Springer, 2026, pp. 26–39. <ExternalAnchor href={ICIAP_DOI_URL}>doi:10.1007/978-3-032-10192-1_3</ExternalAnchor>
            </li>
            <li>
              Morelli N., Marchesini K., Santi D., Grana C., Bolelli F. <em>A New Multicenter Testicular US Dataset and a Lightweight
              Cond-UNet for Generalization in US Segmentation.</em> Proceedings of the British Machine Vision Conference (BMVC), 2026.
            </li>
          </ol>
        </Card>

        <Card icon={FileText} title="Project resources" delay={90}>
          <ul className="space-y-2.5">
            <li>
              <ExternalAnchor href={REPOSITORY_URL}>Source repository (GitHub)</ExternalAnchor>
            </li>
            <li>
              <ExternalAnchor href={REPORT_URL}>Scientific report (PDF)</ExternalAnchor>
            </li>
            <li>
              <a href="/docs" className="inline-flex items-center gap-1 font-medium text-accent-soft underline-offset-2 hover:underline">
                API documentation (OpenAPI)
              </a>
            </li>
          </ul>
          <p className="mt-4 text-[12.5px] text-ink-3">
            The browser talks only to this local service. No analytics, external fonts or third-party requests are made.
          </p>
        </Card>
      </div>
    </section>
  );
}
