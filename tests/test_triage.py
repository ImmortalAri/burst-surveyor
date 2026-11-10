"""Review prioritization must warn without changing numerical exports."""
import copy
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from burstsurveyor.component_review import component_review
from burstsurveyor.triage import assess, publish, ranked, temporal_screen, component_omissions
from burstsurveyor._storage import read, sha256


def packet(name="example", **diagnostic):
    return dict(id=name, status="provisional", reasons=[], flags=[],
                selected=dict(prediction="saved.npz", components=2),
                inspection_status="figure_ready", figure=None,
                diagnostics=dict(flags=[], **diagnostic))


def data_for_screen(extra=False, amplitude=False, negative=False):
    rng = np.random.default_rng(418)
    nf, nt = 64, 256
    t = np.arange(nt)
    model = np.broadcast_to(5*np.exp(-.5*((t-100)/5)**2), (nf, nt)).copy()
    raw = model + rng.normal(size=(nf, nt))
    if extra:
        raw[:32] += 1.0*np.exp(-.5*((t-180)/3)**2)
    if amplitude:
        raw = (np.linspace(.3, 3, nf)[:, None] * model) + rng.normal(size=(nf, nt))
    if negative:
        raw -= 2*np.exp(-.5*((t-102)/1.5)**2)
    e = SimpleNamespace(raw=raw, length=nt, frequency=np.linspace(800, 400, nf),
                        coords=SimpleNamespace(dt=.001),
                        tiles=lambda ids: (ids[i:i+8] for i in range(0, len(ids), 8)))
    data = SimpleNamespace(e=e, indices=np.arange(nf), base=np.zeros(nf), var=np.ones(nf), rho_freq=0., noise_time=(t < 40) | (t > 220))
    return data, model


class TriageTests(unittest.TestCase):
    def test_unavailable_cannot_be_classified_clean(self):
        r = packet()
        r["inspection_status"] = "resource_limited"
        self.assertEqual(assess(r)["disposition"], "unavailable")
        self.assertTrue(assess(r)["requires_review"])

    def test_rank_missing_model_then_temporal_then_spectral_then_routine(self):
        clean, spectral, temporal, unavailable = [packet(n) for n in ("clean", "spectral", "temporal", "unavailable")]
        spectral["diagnostics"]["flags"] = ["frequency_amplitude_structure"]
        temporal["diagnostics"]["component_omissions"] = dict(status="checked", missing=[dict(strength=8.)])
        unavailable["selected"]["prediction"] = None
        rows = ranked([clean, spectral, temporal, unavailable])
        self.assertEqual([r["id"] for r in rows], ["unavailable", "temporal", "clean", "spectral"])

    def test_component_count_is_not_a_warning_by_itself(self):
        r = packet()
        r["selected"]["components"] = 7
        self.assertEqual(assess(r)["priority"], 5)

    def test_separate_faint_pulse_is_reviewed_without_modifying_pixels(self):
        data, model = data_for_screen(extra=True)
        before = model.copy()
        screen = temporal_screen(data, model)
        self.assertTrue(any(w["sign"] == "positive" and w["time_bins"][0] <= 180 < w["time_bins"][1] for w in screen["windows"]))
        self.assertTrue(all(w["frequency_mhz"][0] <= w["frequency_mhz"][1] for w in screen["windows"]))
        np.testing.assert_array_equal(model, before)
        self.assertTrue(any(w.get("location") == "outside_fitted_emission" for w in screen["windows"]))
        r = packet(temporal_screen=screen)
        self.assertEqual(assess(r)["priority"], 5)
        self.assertNotIn("temporal_proposal", screen)

    def test_spectral_amplitude_alone_does_not_request_a_pulse(self):
        data, model = data_for_screen(amplitude=True)
        self.assertEqual(temporal_screen(data, model)["windows"], [])

    def test_negative_trough_is_not_an_add_component_proposal(self):
        data, model = data_for_screen(negative=True)
        screen = temporal_screen(data, model)
        self.assertTrue(any(w["sign"] == "negative" for w in screen["windows"]))
        r = packet(temporal_screen=screen)
        self.assertIn("replicated_negative_residual", [r["code"] for r in assess(r)["information"]])

    def test_width_resolution_and_crop_are_parameter_warnings(self):
        parameters = dict(arrival_time=[.0001, .03], burst_width=[.0001, .04], ref_freq=[600], dm=[0], scattering_timescale=[.01])
        review = component_review(parameters, np.linspace(400, 800, 64), np.arange(100)*.001)
        r = packet(component_review=review)
        codes = {r["code"] for r in assess(r)["information"]}
        self.assertTrue({"sub_bin_width", "broad_component", "crop_sensitive_component"} <= codes)
        self.assertEqual(assess(r)["priority"], 5)

    def test_used_rescue_does_not_clear_persistent_warning(self):
        r = packet()
        r["diagnostics"]["component_omissions"] = dict(status="checked", missing=[dict(strength=8.)])
        r["challenge"] = dict(status="rejected", reason="insufficient improvement")
        self.assertEqual(assess(r)["priority"], 1)
        self.assertIn("rescue_used", [r["code"] for r in assess(r)["reasons"]])

    def test_report_is_deterministic_escaped_and_preserves_assessments(self):
        r = packet('<script>alert(1)</script>')
        r["assessment"] = "review"
        original = copy.deepcopy(r)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            publish(out, [r], {"example": "sealed"})
            first = read(out / "queue-seal.json")
            publish(out, [r], {"example": "sealed"})
            self.assertEqual(first, read(out / "queue-seal.json"))
            self.assertTrue(all(sha256(out / n) == h for n, h in first.items()))
            html = (out / "review.html").read_text(encoding="utf-8")
            self.assertNotIn('<script>alert(1)</script>', html)
            self.assertIn('&lt;script&gt;', html)
            self.assertEqual(read(out / "queue.json")["events"][0]["assessment"], "review")
        self.assertEqual(original, r)

    def test_missing_peak_precedes_in_pulse_shape_information(self):
        missing = packet("missing", component_omissions=dict(status="checked", missing=[dict(strength=4.)]))
        shape = packet("shape", temporal_screen=dict(windows=[dict(sign="positive", strength=9., location="within_fitted_emission")]))
        self.assertEqual(ranked([shape, missing])[0]["id"], "missing")

    def test_strong_separate_peak_is_alerted_on_native_pixels(self):
        data, model = data_for_screen()
        data.e.raw[:32] += 3*np.exp(-.5*((np.arange(256)-180)/3)**2)
        result = component_omissions(data, model, dict(id="example", source_sha256="a"*64))
        self.assertTrue(result["missing"])
        self.assertTrue(any(abs(w["time_bin"]-180)<5 for w in result["missing"]))
        np.testing.assert_array_equal(model, data_for_screen()[1])

    def test_spectral_rescaling_is_not_a_missing_peak(self):
        data, model = data_for_screen(amplitude=True)
        result = component_omissions(data, model, dict(id="example", source_sha256="a"*64))
        self.assertEqual(result["missing"], [])

    def test_two_resolved_peaks_merged_into_one_are_alerted(self):
        data, _ = data_for_screen()
        t = np.arange(256)
        rng = np.random.default_rng(138)
        data.e.raw = (5*np.exp(-.5*((t-100)/3)**2)+5*np.exp(-.5*((t-120)/3)**2))[None, :] + rng.normal(size=(64,256))
        merged = np.broadcast_to(5*np.exp(-.5*((t-110)/12)**2), (64,256)).copy()
        self.assertTrue(component_omissions(data, merged, dict(id="example",source_sha256="a"*64))["missing"])

    def test_width_error_at_existing_peak_is_information_only(self):
        data, _ = data_for_screen()
        broad = np.broadcast_to(5*np.exp(-.5*((np.arange(256)-100)/9)**2), (64,256)).copy()
        self.assertEqual(component_omissions(data, broad, dict(id="example",source_sha256="a"*64))["missing"], [])

    def test_one_bad_channel_does_not_create_an_omission_alert(self):
        data, model = data_for_screen()
        data.e.raw[3,180] += 1000
        self.assertEqual(component_omissions(data, model, dict(id="example",source_sha256="a"*64))["missing"], [])

    def test_unsupported_challenge_does_not_consume_allowance(self):
        from burstsurveyor import assistant
        from burstsurveyor._storage import write
        with tempfile.TemporaryDirectory() as temp:
            ws = Path(temp)
            write(ws / "session.json", dict(mode="combined"))
            with self.assertRaisesRegex(ValueError, "Challenges require bounded"):
                assistant._challenge(ws, None, dict(id="example"), {}, {}, [])
            self.assertFalse((ws / "review/example/challenge.json").exists())

    @unittest.skipUnless(shutil.which("node"), "Node is unavailable for the optional HTML execution check")
    def test_actual_html_search_and_priority_filters_execute(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            publish(out, [dict(packet(), input_name="Example.NPZ"),
                          dict(packet("missing"), input_name="missing.npz", inspection_status="resource_limited")], {})
            html = (out / "review.html").read_text(encoding="utf-8")
            source = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
            program = """
const vm=require('node:vm'),assert=require('node:assert/strict');
const search={value:'',addEventListener(){}},level={value:'5',addEventListener(){}};
const rows=[{id:'event-example',dataset:{name:'Example.NPZ',level:'5'}},
            {id:'event-missing',dataset:{name:'missing.npz',level:'0'}}];
const context={document:{getElementById(id){return id==='search'?search:level},querySelectorAll(){return rows}}};
vm.createContext(context);vm.runInContext(SOURCE,context,{timeout:1000});
context.filter();assert.ok(rows.every(r=>!r.hidden));
search.value='example.npz';context.filter();assert.ok(!rows[0].hidden&&rows[1].hidden);
search.value='';level.value='1';context.filter();assert.ok(rows[0].hidden&&!rows[1].hidden);
""".replace("SOURCE", json.dumps(source))
            subprocess.run([shutil.which("node"), "-e", program], check=True, capture_output=True, timeout=10)


if __name__ == "__main__":
    unittest.main()
