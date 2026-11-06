# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest
import torch

from federatedscope.core.compression.utils import (
    symmetric_uniform_quantization,
    symmetric_uniform_dequantization,
    _symmetric_uniform_quantization,
)
from federatedscope.core.configs.config import global_cfg


class CompressionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_quantization_int4_roundtrip(self):
        conv_w = torch.randn(16, 8, 3, 3)
        fc_w = torch.randn(32, 64)
        fc_b = torch.randn(32)

        state_dict = {
            'conv1.weight': conv_w.clone(),
            'fc1.weight': fc_w.clone(),
            'fc1.bias': fc_b.clone(),
        }

        quant_sd = symmetric_uniform_quantization(state_dict, nbits=4)

        # Packed uint8 verification
        self.assertIn('conv1.weight_quant', quant_sd)
        self.assertIn('conv1.weight_scale', quant_sd)
        self.assertIn('conv1.weight_shape', quant_sd)
        self.assertEqual(quant_sd['conv1.weight_quant'].dtype, torch.uint8)

        # Check payload size reduction: packed bytes should be half the element count
        expected_bytes = (conv_w.numel() + 1) // 2
        self.assertEqual(quant_sd['conv1.weight_quant'].numel(),
                         expected_bytes)

        # Non-weight tensors (like bias) must remain untouched
        self.assertIn('fc1.bias', quant_sd)
        self.assertTrue(torch.equal(quant_sd['fc1.bias'], fc_b))

        # Dequantization reconstruction
        dequant_sd = symmetric_uniform_dequantization(quant_sd)
        self.assertIn('conv1.weight', dequant_sd)
        self.assertIn('fc1.weight', dequant_sd)
        self.assertEqual(dequant_sd['conv1.weight'].shape, conv_w.shape)
        self.assertEqual(dequant_sd['fc1.weight'].shape, fc_w.shape)

        conv_mse = torch.mean((conv_w - dequant_sd['conv1.weight'])**2).item()
        fc_mse = torch.mean((fc_w - dequant_sd['fc1.weight'])**2).item()
        self.assertLess(conv_mse, 0.05)
        self.assertLess(fc_mse, 0.05)

    def test_quantization_odd_element_count_padding(self):
        # Odd number of elements: 5 * 7 = 35 elements
        odd_w = torch.randn(5, 7)
        state_dict = {'fc_odd.weight': odd_w.clone()}

        quant_sd = symmetric_uniform_quantization(state_dict, nbits=4)
        # 35 elements + 1 pad -> 36 nibbles -> 18 bytes
        self.assertEqual(quant_sd['fc_odd.weight_quant'].numel(), 18)

        dequant_sd = symmetric_uniform_dequantization(quant_sd)
        self.assertEqual(dequant_sd['fc_odd.weight'].shape, torch.Size([5, 7]))
        mse = torch.mean((odd_w - dequant_sd['fc_odd.weight'])**2).item()
        self.assertLess(mse, 0.05)

    def test_quantization_zero_tensor(self):
        zero_w = torch.zeros(10, 10)
        state_dict = {'fc_zero.weight': zero_w.clone()}

        quant_sd = symmetric_uniform_quantization(state_dict, nbits=4)
        dequant_sd = symmetric_uniform_dequantization(quant_sd)

        self.assertTrue(torch.equal(dequant_sd['fc_zero.weight'], zero_w))

    def test_quantization_int8_and_int16(self):
        w = torch.randn(10, 20)
        state_dict = {'conv.weight': w.clone()}

        # 8-bit
        quant_8 = symmetric_uniform_quantization(state_dict, nbits=8)
        self.assertEqual(quant_8['conv.weight_quant'].dtype, torch.int8)
        dequant_8 = symmetric_uniform_dequantization(quant_8)
        mse_8 = torch.mean((w - dequant_8['conv.weight'])**2).item()
        self.assertLess(mse_8, 0.01)

        # 16-bit
        quant_16 = symmetric_uniform_quantization(state_dict, nbits=16)
        self.assertEqual(quant_16['conv.weight_quant'].dtype, torch.int16)
        dequant_16 = symmetric_uniform_dequantization(quant_16)
        mse_16 = torch.mean((w - dequant_16['conv.weight'])**2).item()
        self.assertLess(mse_16, 1e-4)

    def test_stochastic_quantization(self):
        x = torch.randn(20, 20)
        qx_det, s_det = _symmetric_uniform_quantization(x,
                                                        nbits=4,
                                                        stochastic=False)
        qx_stoch, s_stoch = _symmetric_uniform_quantization(x,
                                                            nbits=4,
                                                            stochastic=True)

        self.assertEqual(s_det, s_stoch)
        self.assertTrue(torch.all(qx_stoch >= -7))
        self.assertTrue(torch.all(qx_stoch <= 7))

    def test_config_validation(self):
        cfg = global_cfg.clone()
        cfg.quantization.method = 'uniform'

        # Valid 4 bits
        cfg.quantization.nbits = 4
        cfg.assert_cfg()

        # Valid 8 bits
        cfg.quantization.nbits = 8
        cfg.assert_cfg()

        # Valid 16 bits
        cfg.quantization.nbits = 16
        cfg.assert_cfg()

        # Invalid bit width
        cfg.quantization.nbits = 6
        with self.assertRaises(ValueError):
            cfg.assert_cfg()


if __name__ == '__main__':
    unittest.main()
