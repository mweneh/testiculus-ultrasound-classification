import { MODEL_IDS } from './api.js';
import { comparison } from './records.js';

/**
 * Display metadata for the three implemented architectures. Descriptive text
 * only; parameter counts and experiment status come from the frozen record.
 */
const DESCRIPTIONS = {
  custom_cnn: {
    name: 'Custom CNN',
    summary: 'Four-block convolutional network trained from random initialization on grayscale input.',
    training: 'Scratch-trained',
  },
  resnet18: {
    name: 'ResNet18',
    summary: 'Residual network with ImageNet initialization; all layers fine-tuned.',
    training: 'ImageNet-initialized, fully trained',
  },
  mobilenet_v3_small: {
    name: 'MobileNetV3-Small',
    summary: 'Efficient network with frozen ImageNet features and a trained classifier head.',
    training: 'Frozen ImageNet features',
  },
};

export const MODELS = MODEL_IDS.map((id, index) => {
  const record = comparison.models[id];
  return {
    id,
    ...DESCRIPTIONS[id],
    series: index + 1,
    parameters: record.parameters,
    exploratory: record.exploratory,
    status: record.status,
  };
});

export const MODEL_BY_ID = Object.fromEntries(MODELS.map((model) => [model.id, model]));
