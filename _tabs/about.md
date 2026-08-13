---
# the default layout is 'page'
icon: fas fa-info-circle
order: 1
---

<style>
.about-quick-facts {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin: 0.9rem 0 1.5rem;
}
.about-quick-facts .post-tag {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  margin: 0;
}
.about-quick-facts .post-tag i {
  color: var(--link-color);
  font-size: 0.85em;
}

.resume-download {
  margin: 0.6rem 0 0.25rem;
}

.exp-timeline {
  position: relative;
  margin: 1.5rem 0;
  padding-left: 2rem;
}
.exp-timeline::before {
  content: '';
  position: absolute;
  left: 0.4rem;
  top: 0.35rem;
  bottom: 0.35rem;
  width: 2px;
  background: var(--timeline-color);
}
.exp-item {
  position: relative;
  padding-bottom: 1.85rem;
}
.exp-item:last-child {
  padding-bottom: 0;
}
.exp-item::before {
  content: '';
  position: absolute;
  left: -1.6rem;
  top: 0.3rem;
  width: 11px;
  height: 11px;
  border-radius: 50%;
  background: var(--main-bg);
  border: 2px solid var(--timeline-node-bg);
  z-index: 1;
}
.exp-item.is-current::before {
  background: var(--link-color);
  border-color: var(--link-color);
}
.exp-item h3 {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
  margin: 0;
  font-size: 1.05rem;
  color: var(--heading-color);
}
.exp-badge {
  display: inline-block;
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: #fff;
  background: var(--link-color);
  padding: 0.15rem 0.55rem;
  border-radius: 1rem;
}
.exp-meta {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: baseline;
  gap: 0.2rem 1rem;
  margin: 0.3rem 0 0.6rem;
}
.exp-company {
  font-size: 0.9rem;
  color: var(--text-muted-color);
}
.exp-company i {
  color: var(--link-color);
  font-size: 0.85em;
  margin-right: 0.3rem;
}
.exp-company strong {
  color: var(--text-color);
  font-weight: 600;
}
.exp-date {
  font-size: 0.82rem;
  color: var(--text-muted-color);
  white-space: nowrap;
}
.exp-item ul {
  margin: 0;
  padding-left: 1.15rem;
  color: var(--text-color);
  font-size: 0.95rem;
  line-height: 1.6;
}
.exp-item ul li {
  margin-bottom: 0.3rem;
}
.exp-item ul li:last-child {
  margin-bottom: 0;
}
.exp-item p {
  margin: 0;
  color: var(--text-color);
  font-size: 0.95rem;
  line-height: 1.65;
}

.tech-stack {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1.25rem 2rem;
  margin: 1.5rem 0;
}
.tech-label {
  display: block;
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--label-color);
  margin-bottom: 0.5rem;
}
.tech-stack .post-tag {
  display: inline-block;
  margin: 0 0.35rem 0.4rem 0;
}
</style>

Hi there, I'm Sebastià Agramunt Puig, a Software & AI Engineer based in the San Francisco Bay Area (California). I grew up in a small mediterranean town not too far from Barcelona where I went to college. I studied Physics at <a href="https://www.uab.cat/" target="_blank">Universitat Autónoma de Barcelona</a> and continued my academic life with a PhD in theoretical electromagnetism in the same university, working on magnetic levitation with superconductors and magnetic recording with nanoscale magnets (download dissertation <a href="https://www.tdx.cat/handle/10803/129413" target="_blank">here</a>). Along the way I've authored 12 peer-reviewed publications and hold 1 patent — you can find them all on <a href="https://orcid.org/0000-0002-3627-2820" target="_blank">ORCID</a>.

My interests are wide but there are always two common denominators, software development and mathematics. Over more than ten years in industry I've worked on routing algorithms, privacy-preserving machine learning, and high-performance computing, comfortable both leading small teams and working independently. I currently work at <a href="https://www.eikontx.com/" target="_blank">Eikon Therapeutics</a> as a Staff Software Engineer, where I write CUDA algorithms for protein detection and localization in high-throughput drug screening. I'm particularly interested in backend engineering, high-performance computing, CUDA programming, and LLM inference, and I'm always happy to talk about opportunities in those areas.

<div class="about-quick-facts">
  <span class="post-tag"><i class="fa-solid fa-location-dot"></i> Bay Area, CA</span>
  <span class="post-tag"><i class="fa-solid fa-graduation-cap"></i> PhD in Physics</span>
  <span class="post-tag"><i class="fa-solid fa-microchip"></i> CUDA · HPC · LLM inference</span>
  <span class="post-tag"><i class="fa-solid fa-file-lines"></i> 12 publications · 1 patent</span>
</div>

# <i class="fa-solid fa-blog"></i> The blog

Over the years I've been keeping notes about topics I like to learn and have been useful in my professional life. The idea for this blog is to keep a log for myself as well as share with other people that may have similar interests to mine. Please, reach out if you have questions about the content or find errors in the posts.

# <i class="fa-solid fa-file-lines"></i> Resume

<a href="../files/2026.08.12_Sebastia_Agramunt_Puig_Resume.pdf" download class="btn btn-outline-primary btn-sm resume-download"><i class="fa-solid fa-file-arrow-down"></i> Download Resume (PDF)</a>

## Experience

<div class="exp-timeline">
  <div class="exp-item is-current">
    <h3>Staff Software Engineer <span class="exp-badge">Current</span></h3>
    <div class="exp-meta">
      <span class="exp-company"><i class="fa-solid fa-building"></i> <strong>Eikon Therapeutics</strong> · Millbrae, CA</span>
      <span class="exp-date">Jan 2021 – Present</span>
    </div>
    <ul>
      <li>Implemented protein detection and localization algorithms in pure CUDA, with Python bindings and CI/CD for x86 and ARM, achieving a 100x speedup processing 1.5GB movies.</li>
      <li>Contributed to building the data processing pipeline for single-molecule tracking, including image preprocessing algorithms, PostgreSQL database design, and distributed computing.</li>
      <li>Trained and served image segmentation models (U-Net).</li>
      <li>Contributed CI/CD improvements across the company's entire stack and advocated for best practices in testing and artifact publishing.</li>
      <li>Implemented the physics simulation of Brownian motion for proteins in an in-house simulation tool.</li>
    </ul>
  </div>
  <div class="exp-item">
    <h3>Privacy Preserving AI Researcher Contractor (Remote)</h3>
    <div class="exp-meta">
      <span class="exp-company"><i class="fa-solid fa-building"></i> <strong>Neurocat</strong> · Berlin, Germany</span>
      <span class="exp-date">Jan 2021 – Jun 2021</span>
    </div>
    <ul>
      <li>Coauthored a report on the state of the art in privacy-preserving machine learning with Germany's Federal Office for Information Security.</li>
      <li>Focused research on threats associated with transfer learning, including backdoor and adversarial poisoning attacks.</li>
    </ul>
  </div>
  <div class="exp-item">
    <h3>AI Engineer and Privacy Preserving Machine Learning Lead</h3>
    <div class="exp-meta">
      <span class="exp-company"><i class="fa-solid fa-building"></i> <strong>Telefónica Alpha</strong> · Barcelona, Spain</span>
      <span class="exp-date">Nov 2018 – Aug 2020</span>
    </div>
    <ul>
      <li>Initiated and led the company's Privacy Preserving Machine Learning initiative from scratch.</li>
      <li>Acquired a deep understanding of the mathematics behind Differential Privacy, Secure Multi-Party Computation (SMPC), and Fully Homomorphic Encryption (FHE).</li>
      <li>Developed a proof-of-concept for Federated Learning with Secure Aggregation (an SMPC technique) and led a small team to evolve it into a functional mobile product.</li>
      <li>Refactored code to transition a neural collaborative filtering recommender system into production.</li>
    </ul>
  </div>
  <div class="exp-item">
    <h3>Research Scientist, Privacy in Machine Learning</h3>
    <div class="exp-meta">
      <span class="exp-company"><i class="fa-solid fa-building"></i> <strong>OpenMined</strong> · Remote</span>
      <span class="exp-date">Mar 2020 – Aug 2020</span>
    </div>
    <ul>
      <li>Developed an "Introduction to Cryptography" MOOC covering mathematical foundations and Python implementations, reaching over 7,000 students worldwide.</li>
      <li>Researched secure inference on secretly shared machine learning models, focusing on activation function approximation within algebraic rings.</li>
    </ul>
  </div>
  <div class="exp-item">
    <h3>AI & Routing Algorithms Engineer</h3>
    <div class="exp-meta">
      <span class="exp-company"><i class="fa-solid fa-building"></i> <strong>Shotl</strong> · Barcelona, Spain</span>
      <span class="exp-date">Jul 2016 – Oct 2018</span>
    </div>
    <ul>
      <li>Designed and implemented the company's core routing algorithm from scratch, combining simulated annealing, depth-first search, and Dijkstra's algorithm, following an extensive literature review.</li>
      <li>Built fast, reliable software to deploy the routing algorithm in production.</li>
      <li>Analyzed urban demand patterns using machine learning techniques.</li>
      <li>Developed a simulator to evaluate the performance of the routing algorithms.</li>
    </ul>
  </div>
  <div class="exp-item">
    <h3>Data Science Consultant</h3>
    <div class="exp-meta">
      <span class="exp-company"><i class="fa-solid fa-building"></i> <strong>Accenture Digital</strong> · Sant Cugat (Barcelona), Spain</span>
      <span class="exp-date">Jul 2015 – Jul 2016</span>
    </div>
    <ul>
      <li>Utilized ARIMA family models for sales forecasting across various markets for a large cosmetics firm.</li>
      <li>Created and managed a PostgreSQL database, handling data in formats such as CSV and Excel.</li>
      <li>Developed automation scripts in Bash and R integrated with CRON jobs to streamline data loading.</li>
    </ul>
  </div>
  <div class="exp-item">
    <h3>Fellow</h3>
    <div class="exp-meta">
      <span class="exp-company"><i class="fa-solid fa-building"></i> <strong>Data Science Europe</strong> · Dublin, Ireland</span>
      <span class="exp-date">Jan 2015 – Mar 2015</span>
    </div>
    <ul>
      <li>Acquired fundamental data science skills covering SQL, Hive, R, and various machine learning models.</li>
      <li>Analyzed New York City taxi data to predict pickup probabilities by time and location within Manhattan, using Bayesian statistics and Random Forest.</li>
    </ul>
  </div>
  <div class="exp-item">
    <h3>Postdoctoral Researcher</h3>
    <div class="exp-meta">
      <span class="exp-company"><i class="fa-solid fa-building"></i> <strong>Catalan Institute for Nanoscience and Nanotechnology (CIN2)</strong> · Bellaterra, Barcelona, Spain</span>
      <span class="exp-date">Nov 2014 – Dec 2014</span>
    </div>
  </div>
  <div class="exp-item">
    <h3>Postdoctoral Researcher</h3>
    <div class="exp-meta">
      <span class="exp-company"><i class="fa-solid fa-building"></i> <strong>UAB Physics Department</strong> · GNM3, Bellaterra, Barcelona, Spain</span>
      <span class="exp-date">Jul 2013 – Nov 2013</span>
    </div>
  </div>
</div>

## Tech Stack

<div class="tech-stack">
  <div>
    <span class="tech-label">Languages</span>
    <span class="post-tag">C/C++</span><span class="post-tag">CUDA</span><span class="post-tag">Go</span>
  </div>
  <div>
    <span class="tech-label">Scripting</span>
    <span class="post-tag">Python</span><span class="post-tag">Bash</span><span class="post-tag">R</span>
  </div>
  <div>
    <span class="tech-label">ML Frameworks</span>
    <span class="post-tag">PyTorch</span><span class="post-tag">Keras</span><span class="post-tag">scikit-learn</span>
  </div>
  <div>
    <span class="tech-label">GPU Kernels</span>
    <span class="post-tag">CUTLASS</span><span class="post-tag">CuTe DSL</span><span class="post-tag">Triton</span><span class="post-tag">cuBLAS</span><span class="post-tag">cuSOLVER</span>
  </div>
  <div>
    <span class="tech-label">Inference & Serving</span>
    <span class="post-tag">vLLM</span><span class="post-tag">TensorFlow Serving</span>
  </div>
  <div>
    <span class="tech-label">Profiling</span>
    <span class="post-tag">Nsight Compute</span><span class="post-tag">Nsight Systems</span><span class="post-tag">Roofline analysis</span>
  </div>
  <div>
    <span class="tech-label">Data</span>
    <span class="post-tag">NumPy</span><span class="post-tag">Pandas</span><span class="post-tag">Polars</span>
  </div>
  <div>
    <span class="tech-label">Databases</span>
    <span class="post-tag">PostgreSQL</span><span class="post-tag">MongoDB</span><span class="post-tag">Redis</span>
  </div>
  <div>
    <span class="tech-label">Visualization</span>
    <span class="post-tag">Gnuplot</span><span class="post-tag">ggplot2</span><span class="post-tag">Matplotlib</span>
  </div>
  <div>
    <span class="tech-label">MLOps</span>
    <span class="post-tag">Metaflow</span><span class="post-tag">Airflow</span><span class="post-tag">ZenML</span>
  </div>
  <div>
    <span class="tech-label">CI/CD</span>
    <span class="post-tag">GitHub Actions</span><span class="post-tag">Atlassian Bamboo</span><span class="post-tag">Jenkins</span>
  </div>
  <div>
    <span class="tech-label">Other Tools</span>
    <span class="post-tag">Docker</span><span class="post-tag">Git</span><span class="post-tag">pybind11</span>
  </div>
</div>

# <i class="fa-solid fa-envelope"></i> Contact

**Email**: **contact[@]agramunt[dot]me**

If possible, send me your email encrypted with PGP:

**Public Key**: <a href="../files/email_pk.asc" download>email_pk.asc</a>

**Figerprint**: `1DA7 CE68 83F5 02BB AD56  1048 6E73 C83B 19A5 9D3E`

Verify that the key fingerprint is correct with

```bash
 gpg --import email_pk.asc
 gpg --fingerprint paste_my_email_here
```
