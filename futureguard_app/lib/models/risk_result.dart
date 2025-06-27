/// Parsed JSON coming from FastAPI.
///
///  histDates / histBalances  -> blue historical line
///  dates / median / p5 / p95 -> forecast fan-chart
///  best / worst              -> dashed forecast lines
///  insights                  -> 3 short paragraphs
///  recommendations           -> up to 3 bullet points
///
class RiskResult {
  // chart series
  final List<String> histDates;
  final List<double> histBalances;
  final List<String> dates;
  final List<double> median;
  final List<double> p5;
  final List<double> p95;
  final List<double> best;
  final List<double> worst;

  // text blocks
  final List<String> insights;
  final List<String> recommendations;

  RiskResult({
    required this.histDates,
    required this.histBalances,
    required this.dates,
    required this.median,
    required this.p5,
    required this.p95,
    required this.best,
    required this.worst,
    required this.insights,
    required this.recommendations,
  });

  factory RiskResult.fromJson(Map<String, dynamic> j) => RiskResult(
    histDates: List<String>.from(j['hist_dates']),
    histBalances: List<double>.from(j['hist_balances']),
    dates: List<String>.from(j['dates']),
    median: List<double>.from(j['median']),
    p5: List<double>.from(j['p5']),
    p95: List<double>.from(j['p95']),
    best: List<double>.from(j['best']),
    worst: List<double>.from(j['worst']),
    insights: List<String>.from(j['insights']),
    recommendations: List<String>.from(j['recommendations']),
  );
}
