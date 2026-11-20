import { BookOpen, Database, ExternalLink, FileText, ShieldAlert, Split } from 'lucide-react';
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
