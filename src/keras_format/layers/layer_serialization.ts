import {AdvancedActivationLayerSerialization} from './advanced_activation_serialization';
import {DepthwiseConv2DLayerSerialization} from './convolutional_depthwise_serialization';
import {ConvLayerSerialization} from './convolutional_serialization';
import {CoreLayerSerialization} from './core_serialization';
import {MergeLayerSerialization} from './merge_serialization';
import {BatchNormalizationLayerSerialization} from './normalization_serialization';
import {ZeroPadding2DLayerSerialization} from './padding_serialization';
import {PoolingLayerSerialization} from './pooling_serialization';
import {RecurrentLayerSerialization} from './recurrent_serialization';

export type LayerSerialization =
    AdvancedActivationLayerSerialization|DepthwiseConv2DLayerSerialization|
    ConvLayerSerialization|CoreLayerSerialization|MergeLayerSerialization|
    BatchNormalizationLayerSerialization|ZeroPadding2DLayerSerialization|
    PoolingLayerSerialization|RecurrentLayerSerialization;
