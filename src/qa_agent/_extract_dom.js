// DOM extraction script for the QA evidence extractor (extraction.py).
//
// Returns a structured object: { general, images, banners, warnings }.
// Every captured element is tagged with `data-qa-extract-id` so the Python
// side can take a focused screenshot of it by selector afterwards.
//
// Design notes:
//  * We read from the *rendered* DOM (after JS + lazy-load auto-scroll), not raw HTML.
//  * Images are gathered from <img>, <picture>/<source srcset>, srcset/data-* lazy
//    attributes, computed CSS background-image, inline styles and page-builder
//    backgrounds (Elementor / Divi / WPBakery / Fusion / Bricks).
//  * Carousel slides are read straight from the DOM (all slides, not just the
//    first visible one) and marked is_carousel + slide_index.
//  * Nothing is invented — every text field is taken verbatim from the page.

() => {
  const out = { general: {}, images: [], banners: [], warnings: [] };

  const abs = (u) => {
    if (!u) return "";
    u = ("" + u).trim();
    if (!u || u.startsWith("data:")) return u.startsWith("data:") ? "" : "";
    try { return new URL(u, location.href).href; } catch (e) { return u; }
  };
  const clean = (s) => ("" + (s || "")).replace(/\s+/g, " ").trim();
  const vh = window.innerHeight || 900;
  const host = location.hostname;

  let idCounter = 0;
  const mark = (el) => {
    let id = el.getAttribute("data-qa-extract-id");
    if (!id) { id = "qa" + (idCounter++); el.setAttribute("data-qa-extract-id", id); }
    return id;
  };

  const classStr = (el) => {
    try { return (el.className && el.className.toString) ? el.className.toString() : ""; }
    catch (e) { return ""; }
  };

  const ancestorMatch = (el, re) => {
    let n = el;
    for (let i = 0; i < 8 && n; i++) {
      if (re.test(classStr(n)) || re.test(n.id || "") || re.test(n.tagName)) return true;
      n = n.parentElement;
    }
    return false;
  };

  const ctx = (el) => {
    let cs;
    try { cs = getComputedStyle(el); } catch (e) { cs = null; }
    const r = el.getBoundingClientRect();
    const visible = !!cs
      && cs.display !== "none"
      && cs.visibility !== "hidden"
      && parseFloat(cs.opacity || "1") > 0.05
      && r.width > 1 && r.height > 1;
    return { r, visible, cs };
  };

  const bgUrls = (el) => {
    let urls = [];
    try {
      const bg = getComputedStyle(el).backgroundImage;
      if (bg && bg !== "none") {
        const m = bg.match(/url\((['"]?)(.*?)\1\)/g) || [];
        urls = m.map((x) => {
          const mm = x.match(/url\((['"]?)(.*?)\1\)/);
          return mm ? abs(mm[2]) : "";
        }).filter(Boolean);
      }
    } catch (e) { /* ignore */ }
    return urls;
  };

  const sectionHeading = (el) => {
    const sec = el.closest("section, header, article, [class*='section' i], [class*='row' i]");
    if (sec) { const h = sec.querySelector("h1,h2,h3"); if (h) return clean(h.innerText); }
    return "";
  };
  const nearbyText = (el) => {
    const p = el.closest("figure, li, div, section");
    if (p) { const t = clean(p.innerText); if (t) return t.slice(0, 400); }
    return "";
  };
  const linkOf = (el) => { const a = el.closest("a[href]"); return a ? abs(a.getAttribute("href")) : ""; };
  const captionOf = (el) => {
    const f = el.closest("figure");
    if (f) { const c = f.querySelector("figcaption"); if (c) return clean(c.innerText); }
    return "";
  };

  // ---------------- general content ----------------
  try {
    const g = out.general;
    g.page_title = document.title || "";
    const h1 = document.querySelector("h1");
    g.h1 = h1 ? clean(h1.innerText) : "";
    g.headings = Array.from(document.querySelectorAll("h1,h2,h3"))
      .slice(0, 150)
      .map((h) => ({ tag: h.tagName.toLowerCase(), text: clean(h.innerText) }))
      .filter((h) => h.text);

    const bodyText = document.body ? document.body.innerText : "";
    g.raw_text = (bodyText || "").slice(0, 100000);
    g.cleaned_text = clean(bodyText).slice(0, 60000);
    g.main_visible_text = g.cleaned_text;

    const mt = document.querySelector('meta[property="og:title"], meta[name="title"]');
    g.meta_title = mt ? (mt.getAttribute("content") || "") : (document.title || "");
    const md = document.querySelector('meta[name="description"], meta[property="og:description"]');
    g.meta_description = md ? (md.getAttribute("content") || "") : "";
    const canon = document.querySelector('link[rel="canonical"]');
    g.canonical_url = canon ? abs(canon.getAttribute("href")) : "";

    // CTA buttons
    const ctaSel = "a.button, a.btn, button, a[class*='btn' i], a[class*='button' i], "
      + ".elementor-button, .wp-block-button__link, a[role='button'], input[type='submit']";
    const seenC = new Set();
    g.cta_buttons = [];
    document.querySelectorAll(ctaSel).forEach((b) => {
      const t = clean(b.innerText || b.value || b.getAttribute("aria-label") || "");
      if (!t) return;
      const a = b.closest("a[href]");
      const url = a ? abs(a.getAttribute("href")) : "";
      const key = t + "|" + url;
      if (seenC.has(key)) return;
      seenC.add(key);
      g.cta_buttons.push({ text: t, url: url });
    });
    g.cta_buttons = g.cta_buttons.slice(0, 60);

    // links (internal + external)
    const seenL = new Set();
    g.links = [];
    document.querySelectorAll("a[href]").forEach((a) => {
      const href = abs(a.getAttribute("href"));
      if (!href || href.startsWith("javascript:") || href.startsWith("mailto:") || href.startsWith("tel:")) return;
      if (seenL.has(href)) return;
      seenL.add(href);
      let internal = true;
      try { internal = new URL(href).hostname === host; } catch (e) { /* keep true */ }
      g.links.push({ text: clean(a.innerText), href: href, internal: internal });
    });
    g.links = g.links.slice(0, 300);
  } catch (e) {
    out.warnings.push("general content extraction error: " + e.message);
  }

  // ---------------- images ----------------
  const pushImg = (el, sourceType, src, extra) => {
    try {
      const resolved = abs(src);
      if (!resolved && (!extra || !extra.background_image_urls || !extra.background_image_urls.length)) return;
      const c = ctx(el);
      const r = c.r;
      const rec = {
        qa_id: mark(el),
        source_type: sourceType,
        src_url: src || "",
        resolved_url: resolved,
        srcset: (extra && extra.srcset) || "",
        alt_text: (el.getAttribute && (el.getAttribute("alt") || el.getAttribute("aria-label"))) || "",
        title_attribute: (el.getAttribute && el.getAttribute("title")) || "",
        class_name: classStr(el),
        caption: captionOf(el),
        nearby_text: nearbyText(el),
        parent_section_heading: sectionHeading(el),
        linked_url: linkOf(el),
        width: Math.round(r.width),
        height: Math.round(r.height),
        is_visible: c.visible,
        is_above_the_fold: r.top < vh && r.bottom > 0 && c.visible,
        in_header: ancestorMatch(el, /header|masthead|navbar|site-header|topbar/i),
        in_footer: ancestorMatch(el, /footer|site-footer/i),
        in_hero: ancestorMatch(el, /hero|jumbotron|masthead-content|main-banner/i),
      };
      if (extra && extra.background_image_urls) rec.background_image_urls = extra.background_image_urls;
      out.images.push(rec);
    } catch (e) {
      out.warnings.push("image extract error: " + e.message);
    }
  };

  try {
    // <img> (incl. lazy + srcset)
    document.querySelectorAll("img").forEach((img) => {
      const lazy = img.getAttribute("data-src") || img.getAttribute("data-lazy-src")
        || img.getAttribute("data-original") || img.getAttribute("data-lazy") || "";
      const hasSrc = !!img.getAttribute("src");
      const src = img.currentSrc || img.getAttribute("src") || lazy;
      let st = "img";
      if (lazy && !hasSrc) st = "lazy_loaded";
      else if (img.srcset || img.getAttribute("srcset") || img.getAttribute("data-srcset")) st = "srcset";
      pushImg(img, st, src, { srcset: img.getAttribute("srcset") || img.getAttribute("data-srcset") || "" });
    });

    // <picture> / <source srcset>
    document.querySelectorAll("picture source[srcset], source[data-srcset]").forEach((s) => {
      const ss = s.getAttribute("srcset") || s.getAttribute("data-srcset") || "";
      const first = ss.split(",")[0].trim().split(/\s+/)[0];
      if (first) pushImg(s, "picture", first, { srcset: ss });
    });

    // lazy data-* on non-img elements
    document.querySelectorAll("[data-bg],[data-background],[data-background-image],[data-lazy-bg]").forEach((el) => {
      const u = el.getAttribute("data-bg") || el.getAttribute("data-background")
        || el.getAttribute("data-background-image") || el.getAttribute("data-lazy-bg");
      if (u) pushImg(el, "lazy_loaded", u, {});
    });

    // computed CSS background-image / inline style / page-builder backgrounds
    const all = document.querySelectorAll("*");
    let scanned = 0;
    for (let i = 0; i < all.length; i++) {
      if (scanned++ > 6000) { out.warnings.push("background-image scan capped at 6000 elements"); break; }
      const el = all[i];
      if (el.tagName === "IMG" || el.tagName === "SOURCE") continue;
      const urls = bgUrls(el);
      if (!urls.length) continue;
      const styleAttr = el.getAttribute("style") || "";
      const inline = /background/i.test(styleAttr);
      const isBuilder = ancestorMatch(el, /elementor|wpb_|vc_|fl-|et_pb|fusion|brxe|brick|so-widget/i);
      const st = isBuilder ? "page_builder" : (inline ? "inline_style" : "css_background");
      pushImg(el, st, urls[0], { background_image_urls: urls });
    }
  } catch (e) {
    out.warnings.push("image discovery error: " + e.message);
  }

  // ---------------- banners ----------------
  const collectBanner = (el, hint, slideIndex) => {
    try {
      const c = ctx(el);
      const r = c.r;
      if (r.width < 80 || r.height < 40) return;

      const bgs = bgUrls(el);
      const imgs = Array.from(el.querySelectorAll("img"))
        .map((i) => abs(i.currentSrc || i.getAttribute("src") || i.getAttribute("data-src")))
        .filter(Boolean).slice(0, 12);

      const ctaEl = el.querySelector(
        "a.button, a.btn, .elementor-button, .wp-block-button__link, button, a[class*='btn' i], a[role='button']"
      );
      const ctaA = ctaEl ? ctaEl.closest("a[href]") : null;
      const linkA = el.querySelector("a[href]");

      let pos = "mid-page";
      if (el.tagName === "FOOTER" || ancestorMatch(el, /footer/i)) pos = "footer";
      else if (hint === "popup" || ancestorMatch(el, /popup|modal/i)) pos = "popup";
      else if (ancestorMatch(el, /sidebar|widget-area/i)) pos = "sidebar";
      else if (slideIndex != null) pos = "carousel";
      else if (r.top < vh * 0.9) pos = "top-hero";

      out.banners.push({
        qa_id: mark(el),
        banner_type_hint: hint,
        page_position: pos,
        is_above_the_fold: r.top < vh && r.bottom > 0 && c.visible,
        is_visible: c.visible,
        visible_text_html: clean(el.innerText).slice(0, 1500),
        cta_text: ctaEl ? clean(ctaEl.innerText || ctaEl.value || "") : "",
        cta_url: ctaA ? abs(ctaA.getAttribute("href")) : "",
        image_urls: imgs,
        background_image_urls: bgs,
        linked_url: linkA ? abs(linkA.getAttribute("href")) : "",
        width: Math.round(r.width),
        height: Math.round(r.height),
        is_carousel: slideIndex != null,
        slide_index: slideIndex,
      });
    } catch (e) {
      out.warnings.push("banner extract error: " + e.message);
    }
  };

  try {
    const bannerSel = [
      "[class*='hero' i]", "[class*='banner' i]", "[class*='slider' i]",
      "[class*='carousel' i]", "[class*='swiper' i]", "[class*='slick' i]",
      "[class*='owl-carousel' i]", "[class*='promo' i]", "[class*='popup' i]",
      "[class*='modal' i]", "[id*='hero' i]", "[id*='banner' i]",
      // Pricing / purchase widgets: the sticky price block, "Buy Now" /
      // "Enquire Now" CTA cards, cart / checkout panels, money-back guarantee
      // and trust banners. These carry price, discount, guarantee and rating
      // claims that must be QA'd, so treat them as banners.
      "[class*='price' i]", "[id*='price' i]", "[class*='pricing' i]",
      "[id*='pricing' i]", "[class*='cart' i]", "[class*='checkout' i]",
      "[class*='cost' i]", "[id*='cost' i]",
      ".elementor-slides", "section[class*='cta' i]", "footer"
    ].join(",");

    const seen = new Set();
    document.querySelectorAll(bannerSel).forEach((el) => {
      if (seen.has(el)) return;
      const cls = classStr(el) + " " + (el.id || "");
      let hint = "unknown";
      if (/hero/i.test(cls)) hint = "hero";
      else if (/carousel|swiper|slick|owl|slider|elementor-slides/i.test(cls)) hint = "carousel";
      else if (/promo|offer|discount|sale/i.test(cls)) hint = "promotional";
      else if (/price|pricing|cart|checkout|cost/i.test(cls)) hint = "pricing";
      else if (/popup|modal/i.test(cls)) hint = "popup";
      else if (/sidebar/i.test(cls)) hint = "sidebar";
      else if (el.tagName === "FOOTER" || /footer/i.test(cls)) hint = "footer";
      else if (/banner/i.test(cls)) hint = "promotional";

      if (hint === "carousel") {
        const slides = el.querySelectorAll(
          ".swiper-slide, .slick-slide, .owl-item, .elementor-repeater-item, [class*='slide' i]"
        );
        if (slides.length) {
          seen.add(el);
          Array.from(slides).slice(0, 15).forEach((s, i) => { seen.add(s); collectBanner(s, "carousel", i); });
          return;
        }
      }
      seen.add(el);
      collectBanner(el, hint, null);
    });
  } catch (e) {
    out.warnings.push("banner discovery error: " + e.message);
  }

  return out;
}
