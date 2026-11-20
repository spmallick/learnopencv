"""One-time export of the article's exact weights for native C++ inference.

Normal C++ builds and runs do not invoke this authoring tool. The ALIKED
checkpoint is split into a dynamic dense encoder and sparse descriptor head;
the C++ implementation performs the original DKD detection between them.
LightGlue is exported one transformer layer at a time so native C++ retains
the original data-dependent early stopping and point-pruning decisions.

ALIKED mathematics follows Zhao et al.'s BSD-3-Clause implementation,
copyright (c) 2022 Zhao Xiaoming. LightGlue is Apache-2.0 licensed. The
portable deformable-convolution decomposition also follows the Apache-licensed
ALIKED adaptation in fabio-sim/LightGlue-ONNX (whose descriptor code retains
the original BSD notice). See cpp/models/NOTICE.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import types

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from lightglue import ALIKED, LightGlue
from lightglue.aliked import DeformableConv2d

CHECKPOINTS = [
    {'file': 'aliked-n16.pth', 'bytes': 2738091,
     'sha256': '5be8704840ed662d9d8c561bf7279c222092674e7eb05fd0feab94899e9d82f2',
     'url': 'https://github.com/Shiaoming/ALIKED/raw/683d7c65197395c0b3f01ebe76e1084a27e73a65/models/aliked-n16.pth'},
    {'file': 'aliked_lightglue_v0-1_arxiv.pth', 'bytes': 47632827,
     'sha256': 'd975e965b105311a6143194852297dff4f02aea5cc2e10cecfed966ca0e22503',
     'url': 'https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/aliked_lightglue.pth'},
]


def portable_deform(self, tensor):
    """Express the original unmasked 3x3 deformable conv using GridSample."""
    shape = torch._shape_as_tensor(tensor)[-2:].to(tensor)
    maximum = shape.max() / 4.0
    offsets = self.offset_conv(tensor).clamp(-maximum, maximum)
    yy, xx = torch.meshgrid(
        torch.arange(tensor.shape[-2], device=tensor.device, dtype=tensor.dtype),
        torch.arange(tensor.shape[-1], device=tensor.device, dtype=tensor.dtype),
        indexing="ij",
    )
    sampled = []
    for kernel in range(9):
        ky, kx = divmod(kernel, 3)
        y = yy + ky - 1 + offsets[:, 2 * kernel]
        x = xx + kx - 1 + offsets[:, 2 * kernel + 1]
        grid = torch.stack((2*x/(shape[1]-1)-1, 2*y/(shape[0]-1)-1), -1)
        sampled.append(F.grid_sample(tensor, grid, mode="bilinear", align_corners=True))
    values = torch.stack(sampled, 1)
    result = torch.einsum("nkchw,ock->nohw", values, self.regular_conv.weight.flatten(2))
    if self.regular_conv.bias is not None:
        result = result + self.regular_conv.bias.reshape(1, -1, 1, 1)
    return result


class DenseEncoder(nn.Module):
    def __init__(self, extractor):
        super().__init__()
        self.extractor = extractor

    def forward(self, image):
        # Native C++ pads symmetrically to a multiple of 32 before this call.
        m = self.extractor
        x1 = m.block1(image)
        x2 = m.block2(m.pool2(x1))
        x3 = m.block3(m.pool4(x2))
        x4 = m.block4(m.pool4(x3))
        features = torch.cat([
            m.gate(m.conv1(x1)),
            F.interpolate(m.gate(m.conv2(x2)), scale_factor=2, mode="bilinear", align_corners=True),
            F.interpolate(m.gate(m.conv3(x3)), scale_factor=8, mode="bilinear", align_corners=True),
            F.interpolate(m.gate(m.conv4(x4)), scale_factor=32, mode="bilinear", align_corners=True),
        ], 1)
        return F.normalize(features, p=2, dim=1), torch.sigmoid(m.score_head(features))


class DescriptorHead(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, features, keypoints):
        # Batch size is one; N, height and width remain dynamic in ONNX.
        shape = torch._shape_as_tensor(features)
        scale = torch.stack((shape[-1], shape[-2])).to(features) - 1
        pixel = (keypoints / 2 + 0.5) * scale
        integer = pixel.long()
        corners = (integer - 0.5).long()
        x_corner = corners[:, 0].clamp(0, shape[-1] - 4)
        y_corner = corners[:, 1].clamp(0, shape[-2] - 4)
        yy, xx = torch.meshgrid(torch.arange(3, device=features.device),
                                torch.arange(3, device=features.device), indexing="ij")
        x = x_corner[:, None, None] + xx
        y = y_corner[:, None, None] + yy
        linear = (y * shape[-1] + x).flatten()
        patch = torch.index_select(features.flatten(2), 2, linear)
        patch = patch.reshape(128, keypoints.shape[0], 3, 3).permute(1, 0, 2, 3)
        maximum = shape[-2:].to(features).max() / 4.0
        offsets = self.head.offset_conv(patch).clamp(-maximum, maximum)
        offsets = offsets[:, :, 0, 0].reshape(-1, 2, 16).permute(0, 2, 1)
        positions = pixel[:, None] + offsets
        grid = (2 * positions / scale - 1).reshape(1, -1, 1, 2)
        values = F.grid_sample(features, grid, mode="bilinear", align_corners=True)
        values = F.selu(self.head.sf_conv(values)).reshape(128, keypoints.shape[0], 16).permute(1, 0, 2)
        descriptors = torch.einsum("ncp,pcd->nd", values, self.head.agg_weights)
        return F.normalize(descriptors, p=2, dim=1)


class GlueLayer(nn.Module):
    def __init__(self, matcher, index):
        super().__init__()
        self.index = index
        self.input_projection = matcher.input_proj if index == 0 else nn.Identity()
        self.position = matcher.posenc
        self.transformer = matcher.transformers[index]
        self.assignment = matcher.log_assignment[index]
        self.confidence = matcher.token_confidence[index] if index < 8 else None

    def forward(self, descriptors0, descriptors1, keypoints0, keypoints1):
        d0 = self.input_projection(descriptors0)
        d1 = self.input_projection(descriptors1)
        # Fixed-rank reshapes and MatMul avoid legacy ONNX's unknown-rank Einsum
        # inference while using the exact original layer weights and equations.
        def self_attention(x, points):
            m = self.transformer.self_attn
            qkv = m.Wqkv(x).reshape(1, x.shape[1], 4, 64, 3).permute(0, 2, 1, 3, 4)
            q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
            projected = self.position.Wr(points)
            cosine = torch.cos(projected).repeat_interleave(2, -1).unsqueeze(1)
            sine = torch.sin(projected).repeat_interleave(2, -1).unsqueeze(1)
            def rotary(tensor):
                # Positive axes avoid a legacy-exporter rank inference bug for
                # stack(-1) after Gather, which otherwise rotates the wrong axis.
                rotated = torch.cat((-tensor[..., 1::2].unsqueeze(4),
                                      tensor[..., ::2].unsqueeze(4)), 4)
                rotated = rotated.reshape(1, 4, x.shape[1], 64)
                return tensor*cosine + rotated*sine
            q, k = rotary(q), rotary(k)
            attention = torch.softmax(torch.matmul(q, k.transpose(2, 3))*0.125, 3)
            context = torch.matmul(attention, v).transpose(1, 2).reshape(1, x.shape[1], 256)
            return x + m.ffn(torch.cat((x, m.out_proj(context)), -1))
        d0, d1 = self_attention(d0, keypoints0), self_attention(d1, keypoints1)
        m = self.transformer.cross_attn
        def heads(x):
            return x.reshape(1, x.shape[1], 4, 64).transpose(1, 2)
        q0, q1 = heads(m.to_qk(d0))*(m.scale**0.5), heads(m.to_qk(d1))*(m.scale**0.5)
        v0, v1 = heads(m.to_v(d0)), heads(m.to_v(d1))
        similarity = torch.matmul(q0, q1.transpose(2, 3))
        a0, a1 = torch.softmax(similarity, 3), torch.softmax(similarity.transpose(2, 3), 3)
        message0 = m.to_out(torch.matmul(a0, v1).transpose(1, 2).reshape(1, d0.shape[1], 256))
        message1 = m.to_out(torch.matmul(a1, v0).transpose(1, 2).reshape(1, d1.shape[1], 256))
        d0 = d0 + m.ffn(torch.cat((d0, message0), -1))
        d1 = d1 + m.ffn(torch.cat((d1, message1), -1))
        if self.confidence is not None:
            token0, token1 = self.confidence(d0, d1)
        else:
            token0, token1 = torch.ones_like(d0[..., 0]), torch.ones_like(d1[..., 0])
        matchability0 = self.assignment.get_matchability(d0)
        matchability1 = self.assignment.get_matchability(d1)
        z0, z1 = self.assignment.matchability(d0), self.assignment.matchability(d1)
        projected0, projected1 = self.assignment.final_proj(d0)/4, self.assignment.final_proj(d1)/4
        similarity = torch.matmul(projected0, projected1.transpose(1, 2))
        scores = (F.log_softmax(similarity, 2) + F.log_softmax(similarity, 1)
                  + F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2))
        return d0, d1, token0, token1, matchability0, matchability1, scores


def export(module, inputs, output, names_in, names_out, axes):
    module.eval()
    with torch.inference_mode():
        torch.onnx.export(module, inputs, str(output), input_names=names_in,
                          output_names=names_out, dynamic_axes=axes, opset_version=18,
                          dynamo=False, do_constant_folding=True)
    # A readable model file and no implicit external-data sidecar simplify C++ setup.
    import onnx
    model = onnx.load(output)
    onnx.checker.check_model(model)
    onnx.save_model(model, output, save_as_external_data=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    extractor = ALIKED(model_name='aliked-n16', max_num_keypoints=2048).eval()
    original = ALIKED(model_name='aliked-n16', max_num_keypoints=2048).eval()
    for module in extractor.modules():
        if isinstance(module, DeformableConv2d):
            module.forward = types.MethodType(portable_deform, module)
    dense = DenseEncoder(extractor).eval()
    descriptor = DescriptorHead(extractor.desc_head).eval()
    image = torch.rand(1, 3, 96, 128)
    with torch.inference_mode():
        features, scores = dense(image)
    export(dense, (image,), args.output_dir/'aliked_n16_dense.onnx', ['image'], ['features','scores'],
           {'image':{2:'height',3:'width'},'features':{2:'height',3:'width'},'scores':{2:'height',3:'width'}})
    points = torch.rand(37, 2)*1.6-0.8
    export(descriptor, (features, points), args.output_dir/'aliked_n16_descriptor.onnx',
           ['features','keypoints'],['descriptors'],
           {'features':{2:'height',3:'width'},'keypoints':{0:'points'},'descriptors':{0:'points'}})
    matcher = LightGlue(features='aliked', flash=False).eval()
    # Upstream constructors populate Torch's cache. Refuse a changed checkpoint
    # even if its upstream filename stayed the same.
    for checkpoint in CHECKPOINTS:
        payload = (Path(torch.hub.get_dir())/'checkpoints'/checkpoint['file']).read_bytes()
        assert len(payload) == checkpoint['bytes'], checkpoint['file']
        assert hashlib.sha256(payload).hexdigest() == checkpoint['sha256'], checkpoint['file']
    # Explicit attention math is portable across CPU ONNX Runtime builds.
    for module in matcher.modules():
        if hasattr(module, 'has_sdp'):
            module.has_sdp = False
    for index in range(9):
        width = 128 if index == 0 else 256
        wrapper = GlueLayer(matcher, index).eval()
        d0,d1 = torch.rand(1,64,width),torch.rand(1,80,width)
        k0,k1 = torch.rand(1,64,2),torch.rand(1,80,2)
        axes={name:{1:dim} for name,dim in [('descriptors0','n'),('descriptors1','m'),('keypoints0','n'),('keypoints1','m'),('refined0','n'),('refined1','m'),('tokens0','n'),('tokens1','m'),('matchability0','n'),('matchability1','m')]}
        axes['log_scores']={1:'n',2:'m'}
        export(wrapper,(d0,d1,k0,k1),args.output_dir/f'lightglue_layer_{index}.onnx',
               ['descriptors0','descriptors1','keypoints0','keypoints1'],
               ['refined0','refined1','tokens0','tokens1','matchability0','matchability1','log_scores'],axes)
        print('Exported LightGlue layer',index,flush=True)
    if args.validate:
        import onnxruntime as ort
        report = {'torch': torch.__version__, 'onnxruntime': ort.__version__,
                  'platform': platform.platform(), 'comparisons': []}
        def compare(name, actual, expected, tolerance=2e-4):
            error = float(np.max(np.abs(actual - expected.detach().numpy())))
            report['comparisons'].append({'name': name, 'shape': list(actual.shape),
                                          'max_abs_error': error, 'tolerance': tolerance})
            print(name, 'max abs error', error, flush=True)
            assert error < tolerance, (name, error)
        session = ort.InferenceSession(str(args.output_dir/'aliked_n16_dense.onnx'), providers=['CPUExecutionProvider'])
        for height, width in [(96, 128), (128, 160)]:
            test_image = torch.rand(1, 3, height, width)
            with torch.inference_mode():
                expected_f, expected_s = original.extract_dense_map(test_image)
            actual_f, actual_s = session.run(None, {'image': test_image.numpy()})
            compare(f'dense_features_{height}x{width}', actual_f, expected_f)
            compare(f'dense_scores_{height}x{width}', actual_s, expected_s)
        session = ort.InferenceSession(str(args.output_dir/'aliked_n16_descriptor.onnx'),providers=['CPUExecutionProvider'])
        for count in [37, 11]:
            test_points = torch.rand(count, 2)*1.6-0.8
            with torch.inference_mode():
                expected = original.desc_head(features, [test_points])[0][0]
            actual = session.run(None, {'features': features.numpy(), 'keypoints': test_points.numpy()})[0]
            compare(f'descriptor_{count}', actual, expected)
        for index in range(9):
            session = ort.InferenceSession(str(args.output_dir/f'lightglue_layer_{index}.onnx'),
                                           providers=['CPUExecutionProvider'])
            for n, m in [(64, 80), (37, 53)]:
                width = 128 if index == 0 else 256
                d0, d1 = torch.rand(1, n, width), torch.rand(1, m, width)
                k0, k1 = torch.rand(1, n, 2)*2-1, torch.rand(1, m, 2)*2-1
                with torch.inference_mode():
                    p0 = matcher.input_proj(d0) if index == 0 else d0
                    p1 = matcher.input_proj(d1) if index == 0 else d1
                    refined0, refined1 = matcher.transformers[index](
                        p0, p1, matcher.posenc(k0), matcher.posenc(k1))
                    tokens = (matcher.token_confidence[index](refined0, refined1) if index < 8
                              else (torch.ones(1, n), torch.ones(1, m)))
                    assignment = matcher.log_assignment[index]
                    expected = (refined0, refined1, *tokens,
                                assignment.get_matchability(refined0),
                                assignment.get_matchability(refined1),
                                assignment(refined0, refined1)[0][:, :-1, :-1])
                actual = session.run(None, {'descriptors0': d0.numpy(), 'descriptors1': d1.numpy(),
                                            'keypoints0': k0.numpy(), 'keypoints1': k1.numpy()})
                for name, value, target in zip(
                        ['refined0', 'refined1', 'tokens0', 'tokens1', 'matchability0', 'matchability1', 'log_scores'],
                        actual, expected):
                    compare(f'lightglue_{index}_{n}x{m}_{name}', value, target)
        (args.output_dir/'export_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    files=[]
    for path in sorted(args.output_dir.glob('*.onnx')):
        payload=path.read_bytes();files.append({'file':path.name,'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()})
    manifest={'lightglue_commit':'5a650c444718caffc2669dc0094cde908f42dd73',
              'checkpoints': CHECKPOINTS,
              'aliked_variant':'aliked-n16','max_keypoints':2048,'detection_threshold':0.2,'nms_radius':2,
              'lightglue_layers':9,'depth_confidence':0.95,'width_confidence':0.99,'filter_threshold':0.1,
              'onnx_opset':18,'files':files}
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Export complete',sum(x['bytes'] for x in files),flush=True)


if __name__=='__main__':main()
