// script.js

// --- Configurações e Variáveis Globais ---
const API_BASE_URL = "http://127.0.0.1:5000"; // URL base da sua API Flask

// Variáveis para armazenar os dados carregados da API
let todosOsCasos = [];
let todasAsVitimas = [];

// Objeto para manter as instâncias de todos os gráficos.
const chartInstances = {};

// Paleta de cores para os gráficos
const CORES_GRAFICOS = [
  "#1e3a5f",
  "#4a5d7c",
  "#6b82a7",
  "#8b9dba",
  "#a8b9cb",
  "#c5d5dc",
  "#e2f2ee",
];
const CORES_CLUSTERS = ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF"];

// --- Inicialização ---
document.addEventListener("DOMContentLoaded", () => {
  const fim = new Date();
  const inicio = new Date();
  inicio.setFullYear(inicio.getFullYear() - 1);
  document.getElementById("dataInicio").valueAsDate = inicio;
  document.getElementById("dataFim").valueAsDate = fim;

  carregarDadosIniciais();

  document
    .getElementById("dataInicio")
    .addEventListener("change", atualizarDashboard);
  document
    .getElementById("dataFim")
    .addEventListener("change", atualizarDashboard);
  document
    .getElementById("variavelRosca")
    .addEventListener("change", atualizarDashboard);
});

// --- Funções de Carregamento de Dados ---
async function carregarDadosIniciais() {
  try {
    console.log("Iniciando carregamento de dados do backend...");
    const [resCasos, resVitimas] = await Promise.all([
      fetch(`${API_BASE_URL}/api/casos`),
      fetch(`${API_BASE_URL}/api/victims`),
    ]);

    if (!resCasos.ok || !resVitimas.ok) {
      throw new Error(
        `Erro na resposta da API: Casos ${resCasos.status}, Vítimas ${resVitimas.status}`
      );
    }

    todosOsCasos = await resCasos.json();
    todasAsVitimas = await resVitimas.json();

    console.log(
      `Dados carregados: ${todosOsCasos.length} casos, ${todasAsVitimas.length} vítimas.`
    );
    atualizarDashboard();
    carregarDadosDeML();
  } catch (erro) {
    console.error("Falha ao carregar dados iniciais:", erro);
    alert(
      "Não foi possível carregar os dados do servidor. Verifique se o backend está em execução."
    );
  }
}

function carregarDadosDeML() {
  inicializarGraficoModelo();
  carregarDadosBoxplot();
  carregarDadosClustering();
  carregarDadosRegressao();
}

// --- Funções de Atualização e Renderização ---
function atualizarDashboard() {
  if (todosOsCasos.length === 0) {
    console.warn("Não há casos para exibir.");
    return;
  }

  const casosFiltrados = filtrarCasosPorData(todosOsCasos);
  console.log(`${casosFiltrados.length} casos após filtragem por data.`);

  const variavelRosca = document.getElementById("variavelRosca").value;
  atualizarGraficoRosca(casosFiltrados, variavelRosca);
  atualizarGraficoTemporal(casosFiltrados);
  atualizarGraficoVitimas(todasAsVitimas);
}

function filtrarCasosPorData(casos) {
  const dataInicio = document.getElementById("dataInicio").valueAsDate;
  const dataFim = document.getElementById("dataFim").valueAsDate;

  if (dataFim) dataFim.setHours(23, 59, 59, 999);

  return casos.filter((caso) => {
    const dataRef = caso.data_abertura || caso.data_ocorrencia;
    if (!dataRef) return false;

    const dataCaso = new Date(dataRef);
    return (
      (!dataInicio || dataCaso >= dataInicio) &&
      (!dataFim || dataCaso <= dataFim)
    );
  });
}

function contarOcorrencias(array, campo) {
  return array.reduce((acc, item) => {
    const valor = item[campo] || "Não especificado";
    acc[valor] = (acc[valor] || 0) + 1;
    return acc;
  }, {});
}

// --- Funções Específicas de Cada Gráfico ---
function atualizarGraficoRosca(casos, variavel) {
  const contagem = contarOcorrencias(casos, variavel);
  const config = {
    type: "doughnut",
    data: {
      labels: Object.keys(contagem),
      datasets: [
        { data: Object.values(contagem), backgroundColor: CORES_GRAFICOS },
      ],
    },
    options: { responsive: true, maintainAspectRatio: false },
  };
  renderizarGrafico("graficoRosca", config, "graficoRosca");
}

function atualizarGraficoTemporal(casos) {
  const contagemPorMes = casos.reduce((acc, caso) => {
    const dataRef = caso.data_abertura || caso.data_ocorrencia;
    if (dataRef) {
      const mesAno = new Date(dataRef).toISOString().slice(0, 7);
      acc[mesAno] = (acc[mesAno] || 0) + 1;
    }
    return acc;
  }, {});
  const labels = Object.keys(contagemPorMes).sort();
  const data = labels.map((label) => contagemPorMes[label]);
  const config = {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Número de Casos",
          data,
          borderColor: CORES_GRAFICOS[0],
          tension: 0.1,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: { y: { beginAtZero: true } },
    },
  };
  renderizarGrafico("graficoTemporal", config, "graficoTemporal");
}

function atualizarGraficoVitimas(vitimas) {
  const contagemGenero = contarOcorrencias(vitimas, "gender");
  const config = {
    type: "bar",
    data: {
      labels: Object.keys(contagemGenero),
      datasets: [
        {
          label: "Gênero",
          data: Object.values(contagemGenero),
          backgroundColor: CORES_GRAFICOS[1],
        },
      ],
    },
    options: { indexAxis: "y", responsive: true, maintainAspectRatio: false },
  };
  renderizarGrafico("graficoVitimas", config, "graficoVitimas");
}

async function inicializarGraficoModelo() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/modelo/coeficientes`);
    if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    const sortedData = Object.entries(data).sort(
      ([, a], [, b]) => Math.abs(b) - Math.abs(a)
    );
    const config = {
      type: "bar",
      data: {
        labels: sortedData.map(([key]) => key.replace(/_/g, " ")),
        datasets: [
          {
            label: "Importância",
            data: sortedData.map(([, val]) => val),
            backgroundColor: CORES_GRAFICOS[0],
          },
        ],
      },
      options: { indexAxis: "y", responsive: true, maintainAspectRatio: false },
    };
    renderizarGrafico("graficoModelo", config, "graficoModelo");
  } catch (e) {
    console.error("Erro ao carregar coeficientes do modelo:", e);
    document.getElementById(
      "graficoModelo"
    ).innerHTML = `<p style="color:red; text-align:center;">Erro ao carregar gráfico do modelo: ${e.message}</p>`;
  }
}

async function carregarDadosBoxplot() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/casos/estatisticas/boxplot`);
    if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    const config = {
      type: "bar",
      data: {
        labels: Object.keys(data),
        datasets: [
          {
            label: "Idade Mediana",
            data: Object.keys(data).map((tipo) => data[tipo].median),
            backgroundColor: CORES_GRAFICOS[3],
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          tooltip: {
            callbacks: {
              afterLabel: (ctx) => {
                const stats = data[ctx.label];
                return [
                  `Min: ${stats.min.toFixed(1)}`,
                  `Q1: ${stats.q1.toFixed(1)}`,
                  `Q3: ${stats.q3.toFixed(1)}`,
                  `Max: ${stats.max.toFixed(1)}`,
                ];
              },
            },
          },
        },
      },
    };
    renderizarGrafico("graficoBoxplot", config, "graficoBoxplot");
  } catch (e) {
    console.error("Erro ao carregar dados do boxplot:", e);
    document.getElementById(
      "graficoBoxplot"
    ).innerHTML = `<p style="color:red; text-align:center;">Erro ao carregar gráfico de idades: ${e.message}</p>`;
  }
}

async function carregarDadosClustering() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/ml/clustering`);
    if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
    const data = await res.json();
    if (data.error || !Array.isArray(data))
      throw new Error(data.error || "Formato de dados inválido.");

    const datasets = data.map((cluster, i) => ({
      label: `Cluster ${cluster.cluster_id} (${cluster.tamanho} membros)`,
      data: cluster.pontos.map((p) => ({ x: p.x, y: new Date(p.y * 1000) })),
      backgroundColor: CORES_CLUSTERS[i % CORES_CLUSTERS.length],
    }));

    const config = {
      type: "scatter",
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            type: "linear",
            position: "bottom",
            title: { display: true, text: "Idade da Vítima" },
          },
          y: {
            type: "time",
            time: {
              unit: "month",
              tooltipFormat: "dd/MM/yyyy",
              displayFormats: {
                month: "MMM yy",
              },
            },
            title: { display: true, text: "Data do Caso" },
          },
        },
      },
    };
    renderizarGrafico(
      "graficoClustering",
      config,
      "graficoClustering",
      data,
      criarInfoBoxCluster
    );
  } catch (e) {
    console.error("Erro ao carregar dados de clustering:", e);
    document.getElementById(
      "graficoClustering"
    ).innerHTML = `<p style="color:red; text-align:center;">Erro ao carregar gráfico de clustering: ${e.message}</p>`;
  }
}

async function carregarDadosRegressao() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/ml/regressao`);
    if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    const minVal = Math.min(...data.actual, ...data.predicted);
    const maxVal = Math.max(...data.actual, ...data.predicted);

    const config = {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "Real vs. Previsto",
            data: data.actual.map((val, i) => ({
              x: val,
              y: data.predicted[i],
            })),
            backgroundColor: CORES_GRAFICOS[0],
          },
          {
            label: "Linha Ideal",
            data: [
              { x: minVal, y: minVal },
              { x: maxVal, y: maxVal },
            ],
            type: "line",
            borderColor: CORES_CLUSTERS[0],
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: `Modelo de Regressão (R² = ${data.r2.toFixed(3)})`,
          },
        },
        scales: {
          x: { title: { display: true, text: "Idade Real" } },
          y: { title: { display: true, text: "Idade Prevista" } },
        },
      },
    };
    renderizarGrafico(
      "graficoRegressao",
      config,
      "graficoRegressao",
      data,
      criarInfoBoxRegressao
    );
  } catch (e) {
    console.error("Erro ao carregar dados de regressão:", e);
    document.getElementById(
      "graficoRegressao"
    ).innerHTML = `<p style="color:red; text-align:center;">Erro ao carregar gráfico de regressão: ${e.message}</p>`;
  }
}

// --- Funções Auxiliares de Renderização ---
/**
 * Renderiza um gráfico em um container específico, destruindo a instância anterior se existir.
 * @param {string} containerId - O ID do elemento div que conterá o gráfico.
 * @param {object} config - O objeto de configuração do Chart.js.
 * @param {string} chartName - O nome/chave para identificar o gráfico no objeto de controle.
 * @param {object} [infoData] - Dados opcionais para criar uma caixa de informações.
 * @param {function} [infoBoxCreator] - Função opcional para criar a caixa de informações.
 */
function renderizarGrafico(
  containerId,
  config,
  chartName,
  infoData = null,
  infoBoxCreator = null
) {
  const container = document.getElementById(containerId);
  if (!container) return;

  // *** CORREÇÃO CRÍTICA AQUI ***
  // Encontra o parente '.grafico-box' para gerir o info-box corretamente.
  const parentBox = container.closest(".grafico-box");

  // Remove qualquer info-box antiga que exista dentro do .grafico-box
  const oldInfoBox = parentBox.querySelector(".info-box");
  if (oldInfoBox) {
    oldInfoBox.remove();
  }

  // Limpa o container do canvas e cria um novo
  container.innerHTML = "";
  const canvas = document.createElement("canvas");
  container.appendChild(canvas);

  // Destrói o gráfico anterior para evitar vazamentos de memória
  if (chartInstances[chartName]) {
    chartInstances[chartName].destroy();
  }

  // Cria a nova instância do gráfico
  chartInstances[chartName] = new Chart(canvas, config);

  // Se houver dados para uma caixa de info, a cria e adiciona ao .grafico-box
  if (infoData && infoBoxCreator) {
    const infoBox = infoBoxCreator(infoData);
    parentBox.appendChild(infoBox);
  }
}

function criarInfoBoxCluster(data) {
  const infoDiv = document.createElement("div");
  infoDiv.className = "info-box";
  let content = "<h3>Informações dos Clusters</h3>";
  data.forEach((cluster, i) => {
    content += `<p style="color:${
      CORES_CLUSTERS[i % CORES_CLUSTERS.length]
    };"><b>Cluster ${cluster.cluster_id}:</b> ${
      cluster.tamanho
    } membros, idade média de ${cluster.idade_media.toFixed(1)} anos.</p>`;
  });
  infoDiv.innerHTML = content;
  return infoDiv;
}

function criarInfoBoxRegressao(data) {
  const infoDiv = document.createElement("div");
  infoDiv.className = "info-box";
  let content = `<h3>Coeficientes do Modelo (R² = ${data.r2.toFixed(
    3
  )})</h3><ul>`;
  Object.entries(data.coeficientes)
    .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
    .slice(0, 5)
    .forEach(([feat, val]) => {
      content += `<li><b>${feat.replace(/_/g, " ")}:</b> ${val.toFixed(
        3
      )}</li>`;
    });
  content += "</ul>";
  infoDiv.innerHTML = content;
  return infoDiv;
}
