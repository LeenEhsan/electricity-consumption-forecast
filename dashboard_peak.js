document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("peak-form");
  const citySelect = document.getElementById("peak-city");
  let peakChart;

  form.addEventListener("submit", async function (e) {
    e.preventDefault();

    const city = citySelect.value;
    console.log("Fetching peak days for:", city);

    const response = await fetch(`/api/peak_days?city=${city}`);
    const result = await response.json();
    const { dates, values } = result;

    const sortedData = values
      .map((value, i) => ({ value, date: dates[i] }))
      .sort((a, b) => b.value - a.value);

    const sortedDates = sortedData.map((item) => item.date);
    const sortedValues = sortedData.map((item) => item.value);

    const max = Math.max(...sortedValues);
    const min = Math.min(...sortedValues);
    const colors = sortedValues.map((val) => {
      const ratio = (val - min) / (max - min);
      if (ratio < 0.33) return "rgba(255, 223, 70, 0.9)";
      else if (ratio < 0.66) return "rgba(255, 165, 0, 0.9)";
      else return "rgba(255, 0, 0, 0.9)";
    });

    const chartCanvas = document.getElementById("peakChart");
    if (peakChart) peakChart.destroy();

    peakChart = new Chart(chartCanvas, {
      type: "bar",
      data: {
        labels: sortedDates,
        datasets: [
          {
            label: `${city} Peak Consumption (kWh)`,
            data: sortedValues,
            backgroundColor: colors,
            borderRadius: 6,
            borderSkipped: false,
            borderColor: "#ccc",
          },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          tooltip: {
            callbacks: {
              label: (ctx) =>
                `Consumption: ${ctx.parsed.y.toLocaleString()} kWh`,
            },
          },
          legend: {
            labels: {
              color: "#333",
              font: {
                size: 14,
                weight: "bold",
              },
            },
          },
        },
        scales: {
          x: {
            ticks: {
              color: "#333",
            },
            grid: {
              display: false,
            },
          },
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: "Consumption (kWh)",
              color: "#333",
              font: { size: 14 },
            },
            ticks: {
              color: "#333",
            },
          },
        },
      },
    });
  });
});
