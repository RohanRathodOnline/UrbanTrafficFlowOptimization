# Vercel & Cloud Deployment Guide
## Urban Traffic Flow Optimization

This guide outlines how to deploy the **Urban Traffic Flow Optimization** project to **Vercel** and **Streamlit Community Cloud**.

---

## 1. Why Dual Deployment?

* **Vercel Deployment (`public/` + `api/`):**
  * Vercel is designed for serverless functions and ultra-fast static frontend delivery on a global edge CDN.
  * Our dedicated Vercel package includes a high-performance, dark-themed responsive dashboard ([`public/index.html`](file:///r:/UrbanTrafficFlowOptimization/public/index.html), [`public/style.css`](file:///r:/UrbanTrafficFlowOptimization/public/style.css), [`public/app.js`](file:///r:/UrbanTrafficFlowOptimization/public/app.js)) with 12 publication figures, live prediction form, and a serverless Python endpoint in [`api/predict.py`](file:///r:/UrbanTrafficFlowOptimization/api/predict.py).
  * Has zero cold-start delay and runs worldwide with 99.99% uptime.

* **Streamlit Community Cloud Deployment ([`app.py`](file:///r:/UrbanTrafficFlowOptimization/app.py)):**
  * Streamlit requires persistent, bidirectional WebSockets for state and widget reruns.
  * Streamlit Community Cloud ([share.streamlit.io](https://share.streamlit.io)) provides free, dedicated 24/7 container hosting for the full Python GUI.

---

## 2. Deploy to Vercel via CLI (Recommended — 2 Minutes)

Open your terminal in the project root (`R:\UrbanTrafficFlowOptimization`) and run:

```powershell
npx vercel
```

### Step-by-Step Prompt Walkthrough:
1. **`Set up and deploy “R:\UrbanTrafficFlowOptimization”?`** &rarr; Press `Y` and Enter.
2. **`Which scope do you want to deploy to?`** &rarr; Select your personal account or team.
3. **`Link to existing project?`** &rarr; Type `N` and Enter.
4. **`What’s your project’s name?`** &rarr; Keep default `urban-traffic-flow-optimization` or choose a custom name.
5. **`In which directory is your code located?`** &rarr; Keep default `./` (Enter).
6. **`Want to modify settings?`** &rarr; Type `N` (Enter) — our [`vercel.json`](file:///r:/UrbanTrafficFlowOptimization/vercel.json) handles all configuration automatically.

Vercel will output your live URL:
```text
🔍 Inspect: https://vercel.com/YOUR_ACCOUNT/urban-traffic-flow-optimization/...
✅ Production: https://urban-traffic-flow-optimization.vercel.app [copied to clipboard]
```

To deploy subsequent changes directly to production:
```powershell
npx vercel --prod
```

---

## 3. Deploy to Vercel via GitHub Dashboard

1. **Push your repository to GitHub:**
   ```powershell
   git remote add origin https://github.com/YOUR_USERNAME/UrbanTrafficFlowOptimization.git
   git push -u origin main
   ```

2. **Import on Vercel:**
   * Navigate to [vercel.com/new](https://vercel.com/new).
   * Click **Import** next to `UrbanTrafficFlowOptimization`.
   * Keep the Root Directory as `./` and Framework Preset as **Other**.
   * Click **Deploy**.
   * Any future `git push origin main` will automatically trigger a new deployment.

---

## 4. Deploy the Streamlit GUI to Streamlit Community Cloud

If you want to host the original Python Streamlit interface ([`app.py`](file:///r:/UrbanTrafficFlowOptimization/app.py)):

1. Ensure the code is pushed to your GitHub repository.
2. Visit [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
3. Click **New app**.
4. Select your repository: `YOUR_USERNAME/UrbanTrafficFlowOptimization`.
5. Set **Main file path** to: `app.py`.
6. Click **Deploy!**

---

## 5. Local Verification Commands

Before deploying, test the components locally:

* **Preview the Vercel Web Dashboard:**
  ```powershell
  python -m http.server 3000 --directory public
  ```
  Open `http://localhost:3000` in your web browser.

* **Run the Streamlit GUI:**
  ```powershell
  streamlit run app.py
  ```
  Open `http://localhost:8501`.

* **Test CLI Inference:**
  ```powershell
  python main.py --predict "2018-10-15 17:00:00"
  ```
