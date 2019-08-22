# Catalyst info
**Catalyst-info** is a series of posts about the structure of [Catalyst library](https://github.com/catalyst-team/catalyst) and about its [ecosystem](https://github.com/catalyst-team)!

## #1
catalyst-version: `19.08.6` date: `2019-08-22`

### Hello, everyone!

After the release of Catalyst in February it has a lot of new features, which, 
unfortunately, not everyone still knows about. 
I had an idea to post a random fact about a catalyst once in `N` days (it is necessary to think about the periodicity 🤔).
So, the first release of catalyst-info!

---

In Catalyst we all have implemented our favorite Unet's: 
`Unet`, `Linknet`, `FPNUnet`, `PSPnet` and their brothers with resnet-encoders 
`ResnetUnet`, `ResnetLinknet`, `ResnetFPNUnet`, `ResnetPSPnet`. 
Any `Resnet` model can be fitted with any pre-trained encoder (resnet18, resnet34, resnet50, resnet101, resnet152)

Usage
```python
from catalyst.contrib.models.segmentation import ResnetUnet # or any other
model = ResnetUnet(arch="resnet34", pretrained=True)
```
It's easy to load up a `state_dict`
```python
model = ResnetUnet(arch="resnet34", pretrained=False, encoder_params=dict(state_dict="/model/path/resnet34-5c106cde.pth")
```
---

[Link to the model's code.](https://github.com/catalyst-team/catalyst/tree/master/catalyst/contrib/models/segmentation)

All models have a common general structure `encoder-bridge-decoder-head`, 
each of this part can be adjusted separately or even replaced by their own modules!
```python
# In the UnetMetaSpec class
def forward(self, x: torch.Tensor) -> torch.Tensor:
    encoder_features: List[torch.Tensor] = self.encoder(x)
    bridge_features: List[torch.Tensor] = self.bridge(encoder_features)
    decoder_features: List[torch.Tensor] = self.decoder(bridge_features)
    output: torch.Tensor = self.head(decoder_features)
    return output
```

To bolt your model as an encoder for segmentation, you need to inherit it from 
`catalyst.contrib.models.segmentation.encoder.core.EncoderSpec` ([Code](https://github.com/catalyst-team/catalyst/blob/master/catalyst/contrib/models/segmentation/encoder/core.py#L11)).

---

When creating your own block (for any `encoder/bridge/decoder/head`) using the [function](https://github.com/catalyst-team/catalyst/blob/master/catalyst/contrib/models/segmentation/blocks/core.py#L10) `_get_block` 
you can specify the `complexity` parameter, which will create a sequence of [complexity times by](https://github.com/catalyst-team/catalyst/blob/master/catalyst/contrib/models/segmentation/blocks/core.py#L34) `Conv2d + BN + activation`

---

The Upsample part can be specified [either by interpolation or by convolution](https://github.com/catalyst-team/catalyst/blob/febcb66ade07b231348fd8e19614bdd37d548125/catalyst/contrib/models/segmentation/head/unet.py#L18).
