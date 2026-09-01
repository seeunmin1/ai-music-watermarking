import unittest
import sys
import tempfile
import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from c2pa_verifier import verify_c2pa_manifest
from official_adapters import run_official_checks
from audiomark import op_detect
from provenance_core import analyze_provenance
from provider_catalog import load_provider_profiles


@dataclass
class DummyDetection:
    found: bool = False
    record_id: int = 0
    confidence: float = 0.0


class ProvenanceCoreTests(unittest.TestCase):
    def test_google_manifest_without_trusted_signature_is_detected_not_verified(self):
        meta = {
            "manifest": {"claim_generator": "Google SynthID"},
            "fields": {
                "provider": "Google",
                "system": "Gemini v2",
                "created": "2026-08-31T00:00:00Z",
                "unique_id": "gid-1",
            },
            "vendor_hints": ["google", "gemini"],
        }

        result = analyze_provenance(DummyDetection(), meta, [])

        self.assertEqual(result["verdict"], "c2pa_detected_untrusted")
        self.assertEqual(result["generationClaimLevel"], "detected")
        self.assertEqual(result["providerClaimLevel"], "detected")
        self.assertEqual(result["providerVerificationStatus"], "not_verified")
        self.assertEqual(result["attributedProvider"], "Google")
        self.assertEqual(result["provider"], "Google")
        self.assertTrue(any(s["id"] == "google_synthid" and s["status"] == "detected" for s in result["signals"]))

    def test_trusted_c2pa_report_verifies_google_provenance(self):
        meta = {
            "manifest": {"claim_generator": "Google SynthID"},
            "fields": {
                "provider": "Google",
                "system": "Gemini v2",
                "created": "2026-08-31T00:00:00Z",
                "unique_id": "gid-1",
            },
            "vendor_hints": ["google", "gemini"],
        }
        official_results = verify_c2pa_manifest(
            meta,
            load_provider_profiles(),
            c2pa_report={
                "manifest_found": True,
                "signature_valid": True,
                "trusted_signature": True,
                "validation_status": [],
                "tool_status": "ok",
                "tool_available": True,
                "claim_generator": "Google SynthID",
                "fields": {"provider": "Google", "system": "Gemini v2"},
            },
        )

        result = analyze_provenance(DummyDetection(), meta, [], official_results=official_results)

        self.assertEqual(result["verdict"], "ai_generated")
        self.assertEqual(result["claimLevel"], "verified")
        self.assertEqual(result["generationClaimLevel"], "verified")
        self.assertEqual(result["providerVerificationStatus"], "verified")
        self.assertEqual(result["provider"], "Google")
        self.assertEqual(result["attributedProvider"], "Google")
        self.assertEqual(result["attributedProvider"], "Google")
        self.assertEqual(result["resolvedVia"], "Google C2PA Content Credentials")

    def test_google_metadata_hint_does_not_verify_provenance(self):
        meta = {"manifest": None, "fields": None, "vendor_hints": ["gemini"]}

        result = analyze_provenance(DummyDetection(), meta, [])

        self.assertEqual(result["verdict"], "unknown_with_hints")
        self.assertEqual(result["claimLevel"], "unknown")
        self.assertIsNone(result["provider"])
        self.assertTrue(any(s["id"] == "metadata_vendor_hint:google" and s["status"] == "hint" for s in result["signals"]))

    def test_registered_audiomark_watermark_verifies_local_record(self):
        det = DummyDetection(found=True, record_id=0x123456, confidence=88.4)
        registry = [{
            "id_hex": "123456",
            "provider": "Audiomark Labs",
            "system": "DemoTTS",
            "version": "1.0",
            "uid": "am-1",
            "timestamp": "2026-08-31T00:00:00Z",
            "status": "active",
        }]

        result = analyze_provenance(det, {"manifest": None, "fields": None, "vendor_hints": []}, registry)

        self.assertEqual(result["verdict"], "ai_generated")
        self.assertEqual(result["claimLevel"], "verified")
        self.assertEqual(result["provider"], "Audiomark Labs")
        self.assertEqual(result["resolvedVia"], "Audiomark SS/v1")

    def test_classifier_output_is_probable_not_verified(self):
        classifier = {
            "status": "probable",
            "candidates": [{"provider": "ElevenLabs", "confidence": 0.74}],
            "detail": "unit test classifier result",
        }

        result = analyze_provenance(
            DummyDetection(),
            {"manifest": None, "fields": None, "vendor_hints": []},
            [],
            classifier_result=classifier,
        )

        self.assertEqual(result["verdict"], "probably_ai_generated")
        self.assertEqual(result["claimLevel"], "probable")
        self.assertEqual(result["provider"], "ElevenLabs")
        self.assertEqual(result["resolvedVia"], "Local Attribution Classifier")

    def test_official_positive_beats_classifier(self):
        official = [{
            "id": "official:openai",
            "provider": "OpenAI",
            "label": "OpenAI official check",
            "configured": True,
            "verified": True,
            "confidence": 98,
            "system": "OpenAI generated audio",
        }]
        classifier = {
            "status": "probable",
            "candidates": [{"provider": "Suno", "confidence": 0.91}],
        }

        result = analyze_provenance(
            DummyDetection(),
            {"manifest": None, "fields": None, "vendor_hints": []},
            [],
            official_results=official,
            classifier_result=classifier,
        )

        self.assertEqual(result["claimLevel"], "verified")
        self.assertEqual(result["provider"], "OpenAI")
        self.assertEqual(result["resolvedVia"], "OpenAI official check")

    def test_new_provider_profile_enables_trusted_c2pa_match_without_core_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider_path = Path(tmp) / "acme_audio.json"
            provider_path.write_text(json.dumps({
                "provider_id": "acme_audio",
                "display_name": "Acme Audio",
                "aliases": ["acme audio"],
                "products": ["Acme Voice"],
                "supported_media": ["audio/wav"],
                "signals": ["c2pa"],
                "c2pa_issuers": ["Acme Audio, Inc."],
                "official_verification": {"adapter": "c2pa_local", "status": "local"},
                "confidence_policy": {"trusted_c2pa": 96},
                "parser_adapter": "generic_c2pa",
            }), encoding="utf-8")
            profiles = load_provider_profiles(Path(tmp))
            meta = {
                "manifest": {"claim_generator": "Acme Audio, Inc."},
                "fields": {
                    "provider": "Acme Audio",
                    "system": "Acme Voice v1",
                    "unique_id": "acme-1",
                },
                "vendor_hints": [],
            }

            c2pa_results = verify_c2pa_manifest(meta, profiles, c2pa_report={
                "manifest_found": True,
                "signature_valid": True,
                "trusted_signature": True,
                "validation_status": [],
                "tool_status": "ok",
                "tool_available": True,
                "claim_generator": "Acme Audio, Inc.",
            })
            official_results = run_official_checks("sample.wav", meta, profiles)

        self.assertEqual(c2pa_results[0]["provider"], "Acme Audio")
        self.assertTrue(c2pa_results[0]["verified"])
        self.assertFalse(any(r["provider"] == "Acme Audio" and r["verified"] for r in official_results))

    def test_untrusted_c2pa_beats_metadata_hint_but_does_not_verify(self):
        meta = {
            "manifest": {"claim_generator": "Adobe Firefly"},
            "fields": {
                "provider": "Adobe",
                "system": "Adobe Firefly",
                "unique_id": "adobe-1",
            },
            "vendor_hints": ["adobe", "firefly"],
        }
        c2pa_results = verify_c2pa_manifest(meta, load_provider_profiles(), c2pa_report={
            "manifest_found": True,
            "signature_valid": True,
            "trusted_signature": False,
            "validation_status": [{"code": "signingCredential.untrusted"}],
            "tool_status": "ok",
            "tool_available": True,
            "claim_generator": "Adobe Firefly",
        })

        result = analyze_provenance(DummyDetection(), meta, [], official_results=c2pa_results)

        self.assertEqual(result["verdict"], "c2pa_detected_untrusted")
        self.assertEqual(result["attributedProvider"], "Adobe")
        self.assertEqual(result["providerVerificationStatus"], "not_verified")
        self.assertFalse(any(s["provider"] == "Adobe" and s["status"] == "verified" for s in result["signals"]))

    def test_udio_does_not_match_audio_substring(self):
        meta = {
            "manifest": {"claim_generator": "Google SynthID audio fixture"},
            "fields": {
                "provider": "Google",
                "system": "Gemini",
                "unique_id": "gid-2",
            },
            "vendor_hints": ["google", "synthid"],
        }

        result = analyze_provenance(DummyDetection(), meta, [])

        self.assertEqual(result["provider"], "Google")
        self.assertFalse(any(c["provider"] == "Udio" for c in result["providerCandidates"]))

    def test_non_wav_upload_still_runs_provenance_checks(self):
        manifest = {
            "claim_generator": "Google SynthID",
            "assertions": [{
                "label": "ai.statutory_disclosure.ca_sb942",
                "data": {
                    "provider": "Google",
                    "system": "Gemini",
                    "system_version": "test",
                    "created": "2026-08-31T00:00:00Z",
                    "unique_id": "mp3-like-fixture",
                },
            }],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gemini.mp3"
            path.write_bytes(b"ID3\x04\x00\x00fake mp3 c2pa " + json.dumps(manifest).encode("utf-8"))
            result = op_detect(path)

        self.assertEqual(result["provenance"]["generationClaimLevel"], "detected")
        self.assertEqual(result["provenance"]["attributedProvider"], "Google")
        self.assertEqual(result["audio_decode"]["status"], "unsupported")
        self.assertEqual(result["classifier"]["status"], "unsupported")
        self.assertTrue(any("decoded waveform audio" in item for item in result["provenance"]["limitations"]))

    def test_mp3_decode_path_runs_classifier_when_decoder_available(self):
        samples = np.zeros(44100 * 3, dtype=np.float64)
        decode_info = {
            "status": "decoded",
            "codec": "mp3",
            "decoder": "miniaudio",
            "sample_rate": 44100,
            "channels": 2,
            "duration": 3.0,
            "detail": "Audio decoded for watermark detection and local attribution.",
        }
        classifier = {
            "status": "probable",
            "candidates": [{"provider": "Google", "confidence": 0.73}],
            "detail": "unit test classifier result",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gemini.mp3"
            path.write_bytes(b"ID3\x04\x00\x00fake mp3 payload")
            with patch("audiomark._decode_with_miniaudio", return_value=(samples, 44100, decode_info)), \
                 patch("audiomark.classify", return_value=classifier), \
                 patch("audiomark.run_official_checks", return_value=[]):
                result = op_detect(path)

        self.assertEqual(result["audio_decode"]["status"], "decoded")
        self.assertEqual(result["audio_decode"]["codec"], "mp3")
        self.assertEqual(result["classifier"]["status"], "probable")
        self.assertEqual(result["provenance"]["verdict"], "probably_ai_generated")
        self.assertEqual(result["provenance"]["generationClaimLevel"], "probable")
        self.assertEqual(result["provenance"]["attributedProvider"], "Google")
        self.assertEqual(result["provenance"]["providerVerificationStatus"], "not_configured")

    def test_provided_gemini_mp3_reports_decode_status_when_present(self):
        path = Path.home() / "Downloads" / "Cold_Spot_on_the_Floor.mp3"
        if not path.exists():
            self.skipTest("Gemini MP3 sample is not present on this machine.")

        result = op_detect(path)

        self.assertIn(result["audio_decode"]["status"], {"decoded", "unsupported"})
        self.assertIn("classifier", result)
        if result["audio_decode"]["status"] == "unsupported":
            self.assertTrue(any("decoded waveform audio" in item for item in result["provenance"]["limitations"]))


if __name__ == "__main__":
    unittest.main()
