import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';
import { afterEach } from 'vitest';

afterEach(() => cleanup());

if (!URL.createObjectURL) URL.createObjectURL = () => 'blob:preview';
if (!URL.revokeObjectURL) URL.revokeObjectURL = () => {};

// jsdom has no canvas implementation; the heatmap layer treats a null context as "cannot draw".
HTMLCanvasElement.prototype.getContext = () => null;
