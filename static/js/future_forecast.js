window.forecastChart = null;
window.donutChart = null;
let latestForecastJSON = null;

document
  .getElementById("forecast-form")
  .addEventListener("submit", async function (e) {
    e.preventDefault();

    const city = document.getElementById("city").value;
    const duration = parseInt(document.getElementById("duration").value);

    if (!city || !duration) {
      alert("Please select both city and duration.");
      return;
    }

    try {
      const response = await fetch("/api/forecast_future", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ city, duration }),
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Unknown error");

      latestForecastJSON = data;
      document.getElementById("download-json").style.display = "inline-block";

      // Top cards
      document.getElementById(
        "card-cost"
      ).textContent = `${data.metrics.cost} SAR`;
      document.getElementById(
        "card-emissions"
      ).textContent = `${data.metrics.emissions} kg`;
      document.getElementById(
        "card-fuel"
      ).textContent = `${data.metrics.fuel} units`;

      // Forecast Table
      const tbody = document.getElementById("forecast-table-body");
      tbody.innerHTML = "";
      data.results.forEach((d) => {
        const row = `<tr><td class="py-2">${d.date}</td><td class="py-2">${d.predicted_consumption}</td></tr>`;
        tbody.innerHTML += row;
      });
      document
        .getElementById("forecast-table-section")
        .classList.remove("hidden");

      // Line Chart
      // Line Chart
      const ctx = document.getElementById("forecastChart").getContext("2d");
      if (window.forecastChart instanceof Chart) window.forecastChart.destroy();

      // 👇 Calculate dynamic step size
      const min = Math.min(...data.results.map((d) => d.predicted_consumption));
      const max = Math.max(...data.results.map((d) => d.predicted_consumption));
      const range = max - min;
      const step = Math.ceil(range / 5); // adjust number of steps as needed

      window.forecastChart = new Chart(ctx, {
        type: "line",
        data: {
          labels: data.results.map((d) => d.date),
          datasets: [
            {
              label: "Predicted Consumption (kWh)",
              data: data.results.map((d) => d.predicted_consumption),
              borderColor: "#2563eb",
              backgroundColor: "rgba(37,99,235,0.1)",
              borderWidth: 2,
              fill: true,
              tension: 0.4,
              pointRadius: 4,
            },
          ],
        },
        options: {
          responsive: true,
          scales: {
            y: {
              beginAtZero: false,
              ticks: {
                stepSize: step,
                callback: function (value) {
                  return value.toLocaleString(); // format with commas
                },
              },
            },
          },
        },
      });

      // Donut Chart
      const donutCtx = document.getElementById("donutChart").getContext("2d");
      if (window.donutChart instanceof Chart) window.donutChart.destroy();
      window.donutChart = new Chart(donutCtx, {
        type: "doughnut",
        data: {
          labels: ["Cost", "Emissions", "Fuel"],
          datasets: [
            {
              data: [
                data.metrics.cost,
                data.metrics.emissions,
                data.metrics.fuel,
              ],
              backgroundColor: ["#44be4cff", "#f1df57ff", "#fa685e"],
            },
          ],
        },
        options: {
          responsive: true,
          plugins: {
            legend: { position: "bottom" },
          },
        },
      });
    } catch (err) {
      alert("Error occurred: " + err.message);
    }
  });

document.getElementById("download-json").addEventListener("click", () => {
  if (!latestForecastJSON) return;

  const jsonStr = JSON.stringify(latestForecastJSON, null, 2);
  const blob = new Blob([jsonStr], { type: "application/json" });

  // 1. Download locally
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "future_forecast.json";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  // 2. Upload to Flask backend for classification and saving
  const formData = new FormData();
  formData.append("file", blob, "future_forecast.json");

  fetch("/upload_forecast_json", {
    method: "POST",
    body: formData,
  })
    .then((res) => res.json())
    .then((data) => {
      if (data.status === "success") {
        console.log("✅ Uploaded and classified:", data.filename);
      } else {
        console.error("❌ Upload failed:", data);
      }
    })
    .catch((err) => console.error("Upload error:", err));
});
