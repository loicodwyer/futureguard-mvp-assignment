import 'dart:math' as math; // NEW
import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:intl/intl.dart';

class FanChart extends StatelessWidget {
  final List<String> histDates;
  final List<double> histBalances;
  final List<String> dates;
  final List<double> median, p5, p95, best, worst;

  const FanChart({
    super.key,
    required this.histDates,
    required this.histBalances,
    required this.dates,
    required this.median,
    required this.p5,
    required this.p95,
    required this.best,
    required this.worst,
  });

  // helper: "2025-03-15" -> 1 741 996 800 000
  double _dateToX(String iso) =>
      DateTime.parse(iso).millisecondsSinceEpoch.toDouble();

  @override
  Widget build(BuildContext context) {
    // keep just the last 90 days of history
    const keep = 90;
    final skip = histBalances.length > keep ? histBalances.length - keep : 0;
    final histY = histBalances.sublist(skip);
    final histX = histDates.sublist(skip);

    // convert every series to FlSpot lists
    final histSpots = List.generate(
      histY.length,
      (i) => FlSpot(_dateToX(histX[i]), histY[i]),
    );
    final medSpots = List.generate(
      median.length,
      (i) => FlSpot(_dateToX(dates[i]), median[i]),
    );
    final lowSpots = List.generate(
      p5.length,
      (i) => FlSpot(_dateToX(dates[i]), p5[i]),
    );
    final hiSpots = List.generate(
      p95.length,
      (i) => FlSpot(_dateToX(dates[i]), p95[i]),
    );
    final bestSpots = List.generate(
      best.length,
      (i) => FlSpot(_dateToX(dates[i]), best[i]),
    );
    final worstSpots = List.generate(
      worst.length,
      (i) => FlSpot(_dateToX(dates[i]), worst[i]),
    );

    // x-axis span
    final minX = histSpots.first.x;
    final maxX = medSpots.last.x;

    // y-axis span  (rounded to 1 000 euros steps)
    final allY = [...histY, ...p5, ...p95, ...median, ...best, ...worst];
    const step = 1_000.0;
    double _roundDown(double v) => (v / step).floor() * step;
    double _roundUp(double v) => (v / step).ceil() * step;
    final minY = _roundDown(allY.reduce(math.min));
    final maxY = _roundUp(allY.reduce(math.max));

    return Column(
      children: [
        SizedBox(
          height: 280,
          child: LineChart(
            LineChartData(
              minX: minX,
              maxX: maxX,
              minY: minY,
              maxY: maxY,

              // AXES & LABELS
              titlesData: FlTitlesData(
                // left (Y) axis
                leftTitles: AxisTitles(
                  axisNameWidget: const Text('Balance'),
                  sideTitles: SideTitles(
                    showTitles: true,
                    reservedSize: 46,
                    interval: step, // fixed 1 000 € gap
                    getTitlesWidget: (value, meta) {
                      // skip the extreme ticks to avoid crowding
                      if (value == minY || value == maxY) {
                        return const SizedBox.shrink();
                      }
                      // compact “x k” format
                      final sign = value < 0 ? '-' : '';
                      final abs = value.abs();
                      final txt = abs >= 1000
                          ? '${(abs / 1000).toStringAsFixed(abs % 1000 == 0 ? 0 : 1)}k'
                          : abs.toStringAsFixed(0);
                      return Text(
                        '$sign$txt',
                        style: const TextStyle(fontSize: 11),
                      );
                    },
                  ),
                ),
                // bottom (X) axis - only axis name
                bottomTitles: AxisTitles(
                  axisNameWidget: const Text('Date'),
                  sideTitles: SideTitles(showTitles: false),
                ),
                // keep other sides empty
                topTitles: AxisTitles(
                  sideTitles: SideTitles(showTitles: false),
                ),
                rightTitles: AxisTitles(
                  sideTitles: SideTitles(showTitles: false),
                ),
              ),
              gridData: FlGridData(show: false),
              borderData: FlBorderData(show: false),

              // tool-tips
              lineTouchData: LineTouchData(
                enabled: true,
                touchTooltipData: LineTouchTooltipData(
                  tooltipBgColor: Colors.white,
                  tooltipRoundedRadius: 3,
                  getTooltipItems: (spots) => spots.map((t) {
                    final date = DateFormat(
                      'dd MMM yy',
                    ).format(DateTime.fromMillisecondsSinceEpoch(t.x.toInt()));
                    final value = t.y.toStringAsFixed(2);
                    const labels = [
                      'Actual',
                      'P5',
                      'P95',
                      'Median',
                      'Best',
                      'Worst',
                    ];
                    return LineTooltipItem(
                      '$date\n${labels[t.barIndex]}: €$value',
                      const TextStyle(fontSize: 11, height: 1.3),
                    );
                  }).toList(),
                ),
              ),

              // fan-shape fill between P5 and P95
              betweenBarsData: [
                BetweenBarsData(
                  fromIndex: 1,
                  toIndex: 2,
                  color: Colors.orange.withOpacity(.15),
                ),
              ],

              // all six series
              lineBarsData: [
                _line(histSpots, Colors.blue, 2),
                _line(lowSpots, Colors.transparent, 0),
                _line(hiSpots, Colors.transparent, 0),
                _line(medSpots, Colors.orange, 2),
                _line(bestSpots, Colors.green, 1, dash: [6, 4]),
                _line(worstSpots, Colors.red, 1, dash: [6, 4]),
              ],
            ),
          ),
        ),

        const SizedBox(height: 8),

        // legend
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            _legend(color: Colors.blue, text: 'Actual'),
            _legend(color: Colors.orange, text: 'Median'),
            _legend(color: Colors.green, text: 'Best', dashed: true),
            _legend(color: Colors.red, text: 'Worst', dashed: true),
          ],
        ),
      ],
    );
  }

  // single-series helper
  LineChartBarData _line(
    List<FlSpot> s,
    Color c,
    double w, {
    List<int>? dash,
  }) => LineChartBarData(
    spots: s,
    color: c,
    barWidth: w,
    isCurved: false,
    dashArray: dash,
    dotData: FlDotData(show: false),
  );

  // legend helper
  Widget _legend({
    required Color color,
    required String text,
    bool dashed = false,
  }) {
    final bar = dashed
        ? Row(
            children: List.generate(
              4,
              (_) => Container(
                margin: const EdgeInsets.symmetric(horizontal: 1),
                width: 4,
                height: 3,
                color: color,
              ),
            ),
          )
        : Container(width: 18, height: 3, color: color);

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 6),
      child: Row(
        children: [
          bar,
          const SizedBox(width: 4),
          Text(text, style: const TextStyle(fontSize: 11)),
        ],
      ),
    );
  }
}
