// Energy-unit converter. Canonical unit: meV.
(function () {
  'use strict';

  // 1 meV expressed in each target unit.
  var FROM_MEV = {
    meV: 1,
    K:   1 / 0.08617333262,          // 1 meV / k_B(meV/K)   -> K
    Ry:  1 / 13605.693122994,        // meV / Ry(meV)        -> Ry
    cm:  1 / 0.12398419843320,       // meV * (cm^-1 / meV)  -> cm^-1
    THz: 1 / 4.135667696             // meV / h(meV/THz)     -> THz
    // nm handled separately: E[meV] * lambda[nm] = hc = 1.239841984e6
  };
  var HC_MEV_NM = 1.239841984e6;

  function toMeV(value, unit) {
    if (unit === 'nm') return value === 0 ? NaN : HC_MEV_NM / value;
    return value / FROM_MEV[unit];
  }
  function fromMeV(meV, unit) {
    if (unit === 'nm') return meV === 0 ? NaN : HC_MEV_NM / meV;
    return meV * FROM_MEV[unit];
  }

  // 6 significant figures; scientific outside [1e-3, 1e6).
  function fmt(x) {
    if (!isFinite(x)) return '';
    if (x === 0) return '0';
    var abs = Math.abs(x);
    if (abs < 1e-3 || abs >= 1e6) return x.toExponential(5);
    return x.toPrecision(6);
  }

  function parse(str) {
    if (str == null) return NaN;
    var s = String(str).trim().replace(/,/g, '');
    if (s === '') return NaN;
    var n = Number(s);
    return isFinite(n) ? n : NaN;
  }

  function wire() {
    var roots = document.querySelectorAll('.converter[data-converter="energy"]');
    for (var r = 0; r < roots.length; r++) {
      (function (root) {
        var inputs = root.querySelectorAll('input[data-unit]');
        function update(fromInput) {
          var value = parse(fromInput.value);
          if (isNaN(value)) return;
          var meV = toMeV(value, fromInput.getAttribute('data-unit'));
          for (var i = 0; i < inputs.length; i++) {
            var other = inputs[i];
            if (other === fromInput) continue;
            other.value = fmt(fromMeV(meV, other.getAttribute('data-unit')));
          }
        }
        for (var i = 0; i < inputs.length; i++) {
          (function (input) {
            input.addEventListener('input', function () { update(input); });
          })(inputs[i]);
        }
      })(roots[r]);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', wire);
  } else {
    wire();
  }
})();
