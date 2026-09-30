// Free-body diagrams in the report: buttons to switch between the whole
// model and each body, a 3D scene fitted to the size it is shown at, and a
// static image of every view for printing.
(function () {
  "use strict";

  // Printed diagrams: about a page's width at 1:1, drawn at twice that.
  var PRINT_WIDTH = 760;
  var PRINT_HEIGHT = 470;

  function range3(f) {
    return [f(0), f(1), f(2)];
  }

  function max(values) {
    return Math.max.apply(null, values);
  }

  // Axis ranges and aspect ratio (model axes) that fit a 3D scene and its
  // labels into a plot of the given aspect (width / height) and height in
  // pixels. A copy of engmech.report.figure.SceneFit.solve: keep them in step.
  function fitScene(m, aspect, plotHeight) {
    var absUp = m.up.map(Math.abs);
    var absRight = m.right.map(Math.abs);
    function dot(a, b) {
      return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }
    function box(lo, hi) {
      var pad = m.pad * Math.max(max(range3(function (i) { return hi[i] - lo[i]; })), 1e-9);
      lo = lo.map(function (v) { return v - pad; });
      hi = hi.map(function (v) { return v + pad; });
      var span = range3(function (i) { return hi[i] - lo[i]; });
      var big = max(span);
      var ratio = span.map(function (s) { return Math.max(s / big, m.flat); });
      var k = m.fill * Math.min(1 / dot(ratio, absUp), aspect / dot(ratio, absRight));
      return { lo: lo, hi: hi, ratio: ratio.map(function (r) { return r * k; }) };
    }
    var b = box(m.lo, m.hi);
    for (var pass = 0; pass < 2 && m.labels.length; pass++) {
      var span = max(range3(function (i) { return b.hi[i] - b.lo[i]; }));
      var perPx = span / max(b.ratio) / (plotHeight / 2);
      var lo = m.lo.slice();
      var hi = m.hi.slice();
      m.labels.forEach(function (label) {
        label[1].forEach(function (sx) {
          label[2].forEach(function (sy) {
            for (var i = 0; i < 3; i++) {
              var v = label[0][i] + (sx * m.right[i] + sy * m.up[i]) * perPx;
              lo[i] = Math.min(lo[i], v);
              hi[i] = Math.max(hi[i], v);
            }
          });
        });
      });
      b = box(lo, hi);
    }
    return b;
  }

  // The scene settings for a fit, as plotly relayout keys (see
  // engmech.report.figure._apply_scene_fit).
  function sceneSettings(m, b) {
    var out = {};
    var big = max(b.ratio);
    ["xaxis", "yaxis", "zaxis"].forEach(function (name, d) {
      var a = m.shown[d];
      out["scene." + name + ".range"] = m.signs[d] > 0 ? [b.lo[a], b.hi[a]] : [b.hi[a], b.lo[a]];
      out["scene." + name + ".nticks"] = Math.min(9, Math.max(4, Math.round((9 * b.ratio[a]) / big)));
    });
    out["scene.aspectratio"] = { x: b.ratio[m.shown[0]], y: b.ratio[m.shown[1]], z: b.ratio[m.shown[2]] };
    return out;
  }

  // Apply relayout keys like "scene.xaxis.range" to a layout object.
  function assign(layout, settings) {
    Object.keys(settings).forEach(function (path) {
      var keys = path.split(".");
      var node = layout;
      keys.slice(0, -1).forEach(function (k) {
        node = node[k] = node[k] || {};
      });
      node[keys[keys.length - 1]] = settings[path];
    });
    return layout;
  }

  function copy(value) {
    return JSON.parse(JSON.stringify(value));
  }

  // for tests: the fit, to compare with the Python it mirrors
  window.engmechFitScene = fitScene;

  window.engmechFigure = function (containerId, fig, filename) {
    var box = document.getElementById(containerId);
    var gd = box.querySelector(".plot");
    var spec = copy(fig); // plotly adds to the objects it is given
    var menus = spec.layout.updatemenus || [];
    var views = menus.length ? menus[0].buttons : [{ label: "Whole model", args: [{}, {}] }];
    delete spec.layout.updatemenus; // the report has its own buttons, clear of the legend
    var fit = spec.layout.meta && spec.layout.meta.engmech_fit;
    // the live plot takes its size from the page (smaller on phones), not the figure
    var live = copy(spec.layout);
    delete live.width;
    delete live.height;
    if (fit) box.classList.add("is-3d");

    var config = {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"],
      toImageButtonOptions: { format: "png", scale: 2, filename: filename },
    };

    var fitted = "";
    function refit() {
      var size = gd._fullLayout && gd._fullLayout._size;
      if (!fit || !size || !size.w || !size.h) return;
      var key = Math.round(size.w) + "x" + Math.round(size.h);
      if (key === fitted) return;
      fitted = key;
      Plotly.relayout(gd, sceneSettings(fit, fitScene(fit, size.w / size.h, size.h)));
    }

    function showViewButtons() {
      if (views.length < 2) return;
      var bar = box.querySelector(".views");
      var label = document.createElement("span");
      label.className = "views-label";
      label.textContent = "View";
      bar.appendChild(label);
      views.forEach(function (view, k) {
        var button = document.createElement("button");
        button.type = "button";
        button.textContent = view.label.replace(/^Free body: /, "");
        button.title = view.label;
        button.setAttribute("aria-pressed", String(k === 0));
        button.addEventListener("click", function () {
          Plotly.update(gd, view.args[0], view.args[1] || {});
          bar.querySelectorAll("button").forEach(function (b) {
            b.setAttribute("aria-pressed", String(b === button));
          });
        });
        bar.appendChild(button);
      });
      bar.hidden = false;
    }

    // One image per view, one at a time (each 3D image needs a WebGL context).
    function renderPrintViews() {
      var out = box.querySelector(".print-views");
      var images = [];
      return views
        .reduce(function (done, view) {
          return done.then(function () {
            var layout = copy(spec.layout);
            layout.width = PRINT_WIDTH;
            layout.height = PRINT_HEIGHT;
            assign(layout, view.args[1] || {});
            if (fit) {
              var m = layout.margin;
              var w = PRINT_WIDTH - m.l - m.r;
              var h = PRINT_HEIGHT - m.t - m.b;
              assign(layout, sceneSettings(fit, fitScene(fit, w / h, h)));
            }
            var visible = view.args[0].visible;
            var data = spec.data.map(function (trace, i) {
              var t = copy(trace);
              if (visible) t.visible = visible[i];
              return t;
            });
            return Plotly.toImage(
              { data: data, layout: layout },
              { format: "png", width: PRINT_WIDTH, height: PRINT_HEIGHT, scale: 2 }
            ).then(function (url) {
              images.push({ url: url, label: view.label });
            });
          });
        }, Promise.resolve())
        .then(function () {
          images.forEach(function (image) {
            var figure = document.createElement("figure");
            figure.className = "print-view";
            var img = document.createElement("img");
            img.src = image.url;
            img.alt = image.label;
            figure.appendChild(img);
            if (views.length > 1) {
              var caption = document.createElement("figcaption");
              caption.textContent = image.label;
              figure.appendChild(caption);
            }
            out.appendChild(figure);
          });
          box.classList.add("has-print-views"); // print these instead of the live plot
        })
        .catch(function (error) {
          console.warn("engmech: could not draw the diagram for printing", error);
        });
    }

    Plotly.newPlot(gd, copy(spec.data), live, config).then(function () {
      showViewButtons();
      if (fit) {
        refit();
        if (window.ResizeObserver) {
          var timer = null;
          new ResizeObserver(function () {
            clearTimeout(timer);
            timer = setTimeout(refit, 150);
          }).observe(gd);
        }
      }
      return renderPrintViews();
    });
  };

  // Collapsed sections print open, and come back as they were.
  var closed = [];
  window.addEventListener("beforeprint", function () {
    closed = Array.prototype.filter.call(document.querySelectorAll("details"), function (d) {
      return !d.open;
    });
    closed.forEach(function (d) {
      d.open = true;
    });
  });
  window.addEventListener("afterprint", function () {
    closed.forEach(function (d) {
      d.open = false;
    });
    closed = [];
  });
})();
