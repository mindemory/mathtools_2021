﻿import base64, codecs 
sad = '\x72\x6f\x74\x31\x33'
gardo = 'IiIidHJpY2hyb21hY3kyMDE5LnB5ClVzYWdlOgoKZnJvbSB0cmljaHJvbWFjeTIwMTkgaW1wb3J0IGh1bWFuX2NvbG9yX21hdGNoZXIsIGFsdF9odW1hbl9jb2xvcl9tYXRjaGVyCgpZb3Ugc2hvdWxkIGJlIGFibGUgdG8gc2VlIGRvY3VtZW50YXRpb24gYnkgdHlwaW5nIHRoZSBmdW5jdGlvbiBuYW1lIGFuZCBhIHF1ZXN0aW9uIG1hcmsgaW4ganVweXRlciBub3RlYm9vazoKaS5lLgogICAgaHVtYW5fY29sb3JtYXRjaGVyPwogICAgYWx0X2h1bWFuX2NvbG9yX21hdGNoZXI/CgoiIiIKCgppbXBvcnQgbnVtcHkgYXMgbnAKCmFsbF9jb25lcyA9IG5wLmFycmF5KFtbNC4yNTIwMDBlLTAzLCA4LjY2MTAwMGUtMDMsIDEuNTkwNjAwZS0wMiwgMi4zNDY1MDBlLTAyLAogICAgICAgICAgICAgICAgICAgMy4wMjM2MDBlLTAyLCAzLjQ0ODgwMGUtMDIsIDQuMTQxNzAwZS0wMiwgNi4yODM1MDBlLTAyLAogICAgICAgICAgICAgICAgICAgMS4wMjM2MjBlLTAxLCAxLjYyNTIwMGUtMDEsIDIuNjM3ODAwZS0wMSwgNC4yNDU2NzBlLTAxLAogICAgICAgICAgICAgICAgICAgNi4xODg5ODBlLTAxLCA3Ljc1NzQ4MGUtMDEsIDguODY0NTcwZS0wMSwgOS41NzE2NTBlLTAxLAogICAgICAgICAgICAgICAgICAgOS45NjY5MzBlLTAxLCAxLjAwMDc4N2UrMDAsIDkuNjc4NzQwZS0wMSwgOC45NzE2NTBlLTAxLAogICAgICAgICAgICAgICAgICAgNy45NzMyMzBlLTAxLCA2LjcyNTk4MGUtMDEsIDUuMzE4MTEwZS0wMSwgMy44MTI2MDBlLTAxLAogICAgICAgICAgICAgICAgICAgMi41NzMyMzBlLTAxLCAxLjU5Njg1MGUtMDEsIDkuMTY1NDAwZS0wMiwgNC44MzQ3MDBlLTAyLAogICAgICAgICAgICAgICAgICAgMi41ODI3MDBlLTAyLCAxLjI0NDEwMGUtMDIsIDYuMjk5MDAwZS0wM10sCiAgICAgICAgICAgICAgICAgIFs0LjYwMjAwMGUtMDMsIDkuNzE2MDAwZS0wMywgMS44OTIxMDBlLTAyLCAzLjE3MDUwMGUtMDIsCiAgICAgICAgICAgICAgICAgICA0Ljc4MTQwMGUtMDIsIDYuMzY2NzAwZS0wMiwgOC42MTY3MDBlLTAyLCAxLjMwNjU3MGUtMDEsCiAgICAgICAgICAgICAgICAgICAxLjg5MjEwMGUtMDEsIDIuNjc3MDYwZS0wMSwgMy45NzU5NzBlLTAxLCA1Ljk2Nzc4MGUtMDEsCiAgICAgICAgICAgICAgICAgICA4LjEwNTM0MGUtMDEsIDkuNDQ1MTUwZS0wMSwgMS4wMDAwMDBlKzAwLCA5Ljg5NzcyMGUtMDEsCiAgICAgICAgICAgICAgICAgICA5LjI1ODUwMGUtMDEsIDguMDkwMDAwZS0wMSwgNi41MzAzMDBlLTAxLCA0Ljc4NjUwMGUtMDEsCiAgICAgICAgICAgICAgICAgICAzLjE4ODQ0MGUtMDEsIDEuOTQwNjgwZS0wMSwgMS4xMDQ1ODBlLTAxLCA1Ljg1NTMwMGUtMDIsCiAgICAgICAgICAgICAgICAgICAyLjk2NjAwMGUtMDIsIDEuNDMxOTAwZS0wMiwgNy4xNTkwMDBlLTAzLCAzLjMyNDAwMGUtMDMsCiAgICAgICAgICAgICAgICAgICAxLjUzNDAwMGUtMDMsIDcuNjcwMDAwZS0wNCwgMi41NjAwMDBlLTA0XSwKICAgICAgICAgICAgICA'
tram = 'tVPNtJmRhAmD0ZGxjMF0jZFjtZl42AQZ0ZGOyYGNkYPN2YwLlAmxkZTHgZQRfVQxhZQL5AmpjMF0jZFjXVPNtVPNtVPNtVPNtVPNtVPNtVQRhZQNjZQNjMFfjZPjtBF4kBQLjAGOyYGNkYPN4YwNlZmV2ZTHgZQRfVQLhBGZ3BGtjMF0jZFjXVPNtVPNtVPNtVPNtVPNtVPNtVQDhAwt5BGVjMF0jZFjtZv43BGN3ZQOyYGNkYPNkYwL2AwL3ZTHgZQRfVQxhAwt5BGNjMF0jZvjXVPNtVPNtVPNtVPNtVPNtVPNtVQDhAwHkZwNjMF0jZvjtZv4mZwH2ZQOyYGNlYPNkYwR2ZwtjZTHgZQVfVQZhBQp2ZQNjMF0jZljXVPNtVPNtVPNtVPNtVPNtVPNtVQZhBQp2ZQNjMF0jZljtZP4jZQNjZQOyXmNjYPNjYwNjZQNjZTHeZQNfVQNhZQNjZQNjMFfjZPjXVPNtVPNtVPNtVPNtVPNtVPNtVQNhZQNjZQNjMFfjZPjtZP4jZQNjZQOyXmNjYPNjYwNjZQNjZTHeZQNfVQNhZQNjZQNjMFfjZPjXVPNtVPNtVPNtVPNtVPNtVPNtVQNhZQNjZQNjMFfjZPjtZP4jZQNjZQOyXmNjYPNjYwNjZQNjZTHeZQNfVQNhZQNjZQNjMFfjZPjXVPNtVPNtVPNtVPNtVPNtVPNtVQNhZQNjZQNjMFfjZPjtZP4jZQNjZQOyXmNjYPNjYwNjZQNjZTHeZQOqKFxXPtcxMJLtnUIgLJ5sL29fo3WsoJS0L2uypvu0MKA0K2kcM2u0YPOjpzygLKWcMKZcBtbtVPNtVvVvVSAcoKIfLKEyVTRtpTSlqTywqJkupvObqJ1uovOiLaAypaMypvNbMJy0nTIlVTuyLJk0nUxfVT9lVUqbMJ4tLJk0VTymVUAyqPO0olOHpaIyYPOwo2kipvO2nKAco24tMTIznJAcMJ50XFOcovOuPvNtVPNtVPNtL29fo3VgoJS0L2ucozptMKujMKWcoJIhqP4XPvNtVPNtVPNtFTIuoUEbrFOiLaAypaMypwbXVPNtVPNtVPNtVPNtn25iLy9mMKE0nJ5aplN9VTu1oJShK2AioT9lK21uqTAbMKVbqTImqS9fnJqbqPjtpUWcoJSlnJImXDbXVPNtVRSlM3Z6PvNtVPNtVPNtqTImqS9fnJqbqQbtLFNmZF1xnJ1yoaAco25uoPOwo2k1oJ4tqzIwqT9lYPOwo250LJyhnJ5aVUEbMFOmpTIwqUWuoPOxnKA0pzyvqKEco24to2LtLFO0MKA0VTkcM2u0YPO3nKEbPvNtVPNtVPNtVPNtVUquqzIfMJ5aqTumVUAuoKOfMJDtMaWioFN0ZQNtqT8tAmNjVT5gVTyhVQRjoz0tnJ5wpzIgMJ50pl4tVRSfqTIlozS0nKMyoUxfVTy0VTAuovOvMFOuVQZkYKWiqlOgLKElnKtfVUqcqTtXVPNtVPNtVPNtVPNtMJSwnPOwo2k1oJ4tL29hqTScozyhMlOuVUEyp3DtoTyanUDhPvNtVPNtVPNtpUWcoJSlnJImBvOmnT91oTDtLzHtLFNmZF1lo3ptoJS0pzy4VTAioaEunJ5cozptqTuyVUOlnJ1upaxtoTyanUEmVUEbLKDtqTuyVTu1oJShVT11p3DtoJy4VTyhVT9lMTIlVUEiVT1uqTAbPvNtVPNtVPNtVPNtVUEbMFOupUOyLKWuozAyVT9zVUEbMFO0MKA0VTkcM2u0XUZcYvNtGz9loJSfoUxfVUEbnKZtq291oTDtL29hqTScovNmVUOlnJ1upzyyplNbZlOwo2k1oJ5mXFjXVPNtVPNtVPNtVPNtLaI0VUEbMFOzqJ5wqTyiotbtVPNtVPNtVPNtVPO3nJkfVUOlo2E1L2HtLKOjpz9jpzyuqTHtpzImpT9hp2ImVTMipvOuoaxtoaIgLzIlVT9zpUWcoJSlnJImYt'
krugz = 'ogICAgUmV0dXJuczoKICAgICAgICBrbm9iX3NldHRpbmdzOiB0aGUgaHVtYW4gcmVzcG9uc2U6IGEgdmVjdG9yIGNvbnRhaW5pbmcgdGhlIGludGVuc2l0aWVzIG9mIHRoZSBwcmltYXJ5IGxpZ2h0cyB0aGF0IHRoZSBodW1hbgogICAgICAgICAgICBjaG9vc2VzIHRvIGJlc3QgbWF0Y2ggdGhlIHRlc3QgbGlnaHQocykuCgogICAgIiIiCiAgICBhc3NlcnQgdGVzdF9saWdodC5zaGFwZVswXSA9PSAzMQogICAgaWYgdGVzdF9saWdodC5uZGltID09IDE6ICAjIHR1cm4gaW50byBjb2wtdmVjCiAgICAgICAgdGVzdF9saWdodCA9IHRlc3RfbGlnaHRbOiwgTm9uZV0KCiAgICBudW1fdG90YWxfY29uZXMgPSAzCgogICAgY29uZXMgPSBhbGxfY29uZXMKCiAgICBbdXUsIHNzLCB2dF0gPSBucC5saW5hbGcuc3ZkKGNvbmVzIEAgcHJpbWFyaWVzKQogICAgdnYgPSB2dC5UCgogICAgbWF0Y2hfaW5kID0gW2lpIGZvciBpaSBpbiByYW5nZShsZW4oc3MpKSBpZiBucC5hYnMoc3NbaWldKSA+IChucC5hYnMoc3NbMF0pIC8gMWU2KV0KCiAgICBpbnYgPSB2dls6LCBtYXRjaF9pbmRdIEAgbnAubGluYWxnLmludihucC5kaWFnKHNzW21hdGNoX2luZF0pKSBAIHV1WzosIG1hdGNoX2luZF0uVAogICAgbnVsbCA9IHZ2WzosIG5wLnNldGRpZmYxZChucC5hcmFuZ2UobnVtX3RvdGFsX2NvbmVzKSwgbWF0Y2hfaW5kKV0KICAgIGxpZ2h0X3Jlc3BvbnNlID0gY29uZXMgQCB0ZXN0X2xpZ2h0CiAgICBtZWFuX3Jlc3BvbnNlID0gbnAubWVhbihsaWdodF9yZXNwb25zZSkKICAgIHN0ZF9yZXNwb25zZSA9IG5wLnN0ZChsaWdodF9yZXNwb25zZSkKCiAgICBrbm9iX3NldHRpbmdzID0gaW52IEAgbGlnaHRfcmVzcG9uc2UKCiAgICBpZiBudWxsLnNoYXBlWzFdID4gMDoKICAgICAgICBwcmludCgnYWx0X2tub2JzOlxuJykKICAgICAgICBrbm9iX3NldHRpbmdzICs9IG51bGwgQCAobWVhbl9yZXNwb25zZSArIDAuNSpzdGRfcmVzcG9uc2UqbnAucmFuZG9tLnJhbmRuKDEsIHRlc3RfbGlnaHQuc2hhcGVbMV0pKQogICAgcmV0dXJuIGtub2Jfc2V0dGluZ3MKCgpkZWYgYWx0X2h1bWFuX2NvbG9yX21hdGNoZXIodGVzdF9saWdodCwgcHJpbWFyaWVzKToKICAgICIiIiBTaW11bGF0ZSBhIHBhcnRpY3VsYXIgaHVtYW4gb2JzZXJ2ZXIgKGVpdGhlciBoZWFsdGh5LCBvciB3aGVuIGFsdCBpcyBzZXQgdG8gVHJ1ZSwgY29sb3IgdmlzaW9uIGRlZmljaWVudCkgaW4gYQogICAgICAgIGNvbG9yLW1hdGNoaW5nIGV4cGVyaW1lbnQuCgogICAgICAgIEV4YW1wbGUgdXNhZ2U6CiAgICAgICAgICAgIGtub2Jfc2V0dGluZ3MgPSBhbHRfaHVtYW5fY29sb3JfbWF0Y2hlcih0ZXN0X2xpZ2h0LCBwcmltYXJpZXMpCgogICAgQXJnczoKICAgICAgICB0ZXN0X2xpZ2h0OiBhIDMxLWRpbWVuc2lvbmFsIGNvbHVtbiB2ZWN0b3IsIGNvbnRhaW5pbmcgdGhlIHNwZWN0cmFsIGRpc3RyaWJ1dGlvbiBvZiBhIHRlc3QgbGlnaHQsIHdpdGgKICAgICAgICAgICAgd2F2ZWxlbmd0a'
missy = 'UZtp2SgpTkyMPOzpz9gVQDjZPO0olN3ZQNtoz0tnJ4tZGOhoFOcozAlMJ1yoaEmYvNtDJk0MKWhLKEcqzIfrFjtnKDtL2ShVTWyVTRtZmRgpz93VT1uqUWcrPjtq2y0nNbtVPNtVPNtVPNtVPOyLJAbVTAioUIgovOwo250LJyhnJ5aVTRtqTImqPOfnJqbqP4XVPNtVPNtVPOjpzygLKWcMKZ6VUAbo3IfMPOvMFOuVQZkYKWiqlOgLKElnKttL29hqTScozyhMlO0nTHtpUWcoJSlrFOfnJqbqUZtqTuuqPO0nTHtnUIgLJ4toKImqPOgnKttnJ4to3WxMKVtqT8toJS0L2tXVPNtVPNtVPNtVPNtqTuyVTSjpTIupzShL2Hto2LtqTuyVUEyp3DtoTyanUDbplxhVPOBo3WgLJkfrFjtqTucplO3o3IfMPOwo250LJyhVQZtpUWcoJSlnJImVPtmVTAioUIgoaZcYNbtVPNtVPNtVPNtVPOvqKDtqTuyVTM1ozA0nJ9hPvNtVPNtVPNtVPNtVUqcoTjtpUWiMUIwMFOupUOlo3OlnJS0MFOlMKAjo25mMKZtMz9lVTShrFOhqJ1vMKVto2MjpzygLKWcMKZhPvNtVPOFMKE1pz5mBtbtVPNtVPNtVTgho2Wsp2I0qTyhM3Z6VUEbMFObqJ1uovOlMKAjo25mMGbtLFO2MJA0o3VtL29hqTScozyhMlO0nTHtnJ50MJ5mnKEcMKZto2LtqTuyVUOlnJ1upaxtoTyanUEmVUEbLKDtqTuyVTu1oJShPvNtVPNtVPNtVPNtVTAbo29mMKZtqT8tLzImqPOgLKEwnPO0nTHtqTImqPOfnJqbqPumXF4XPvNtVPNvVvVXVPNtVTSmp2IlqPO0MKA0K2kcM2u0YaAbLKOyJmOqVQ09VQZkPvNtVPOcMvO0MKA0K2kcM2u0Yz5xnJ0tCG0tZGbtVPZtqUIlovOcoaEiVTAioP12MJZXVPNtVPNtVPO0MKA0K2kcM2u0VQ0tqTImqS9fnJqbqSf6YPOBo25yKDbXVPNtVT51oI90o3EuoS9wo25yplN9VQZXVPNtVTAiozImVQ0tLJkfK2AiozImJltjYQVcYQcqPtbtVPNtJ3I1YPOmpljtqaEqVQ0toaNhoTyhLJkaYaA2MPuwo25yplONVUOlnJ1upzyyplxXVPNtVUM2VQ0tqaDhINbXVPNtVT1uqTAbK2yhMPN9VSgcnFOzo3VtnJxtnJ4tpzShM2HboTIhXUAmXFxtnJLtoaNhLJWmXUAmJ2ycKFxtCvNboaNhLJWmXUAmJmOqXFNiVQSyAvyqPtbtVPNtnJ52VQ0tqaMoBvjtoJS0L2usnJ5xKFONVT5jYzkcozSfMl5coaLboaNhMTyuMlump1ggLKEwnS9cozEqXFxtDPO1qIf6YPOgLKEwnS9cozEqYyDXVPNtVT51oTjtCFO2qyf6YPOhpP5mMKExnJMzZJDboaNhLKWuozqyXT51oI90o3EuoS9wo25yplxfVT1uqTAbK2yhMPyqPvNtVPOfnJqbqS9lMKAjo25mMFN9VTAiozImVRNtqTImqS9fnJqbqNbtVPNtoJIuoy9lMKAjo25mMFN9VT5jYz1yLJ4boTyanUEspzImpT9hp2HcPvNtVPOmqTEspzImpT9hp2HtCFOhpP5mqTDboTyanUEspzImpT9hp2HcPtbtVPNtn25iLy9mMKE0nJ5aplN9VTyhqvONVTkcM2u0K3Wyp3OioaAyPtbtVPNtnJLtoaIfoP5mnTSjMIfkKFN+VQN6PvNtVPNtVPNtn25iLy9mMKE0nJ5aplNeCFOhqJkfVRNtXT1yLJ5spzImpT9hp2HtXlNjYwHdp3ExK3Wyp3OioaAyXz5jYaWuozEioF5lLJ5xovtkYPO0MKA0K2kcM2u0YaAbLKOyJmSqXFxXVPNtVUWyqUIlovOeoz9vK3AyqUEcozqmPtb='
ryndo = eval('\x67\x61\x72\x64\x6f') + eval('\x63\x6f\x64\x65\x63\x73\x2e\x64\x65\x63\x6f\x64\x65\x28\x74\x72\x61\x6d\x2c\x73\x61\x64\x29')+ eval('\x6b\x72\x75\x67\x7a') + eval('\x63\x6f\x64\x65\x63\x73\x2e\x64\x65\x63\x6f\x64\x65\x28\x6d\x69\x73\x73\x79\x2c\x73\x61\x64\x29')
eval(compile(base64.b64decode(eval('\x72\x79\x6e\x64\x6f')),'<string>','exec'))