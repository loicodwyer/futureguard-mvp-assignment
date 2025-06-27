import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';

class BalanceChart extends StatelessWidget {
  const BalanceChart({
    super.key,
    required this.hist,
    required this.median,
    required this.p5,
    required this.p95,
  });

  final List<double> hist;
  final List<double> median;
  final List<double> p5;
  final List<double> p95;

  // helper to turn value-lists into FlSpot lists
  List<FlSpot> _spots(List<double> vals, int offset) => List.generate(
    vals.length,
    (i) => FlSpot((i + offset).toDouble(), vals[i]),
  );

  @override
  Widget build(BuildContext context) {
    // X-axis: 0..hist.length-1       => historic part
    //         hist.length..end       => forecast part
    final h = hist.length;
    return SizedBox(
      height: 240,
      child: LineChart(
        LineChartData(
          minY:
              [hist, p5].expand((x) => x).reduce((a, b) => a < b ? a : b) *
              1.05,
          maxY:
              [hist, p95].expand((x) => x).reduce((a, b) => a > b ? a : b) *
              1.05,
          lineTouchData: LineTouchData(enabled: false),
          titlesData: FlTitlesData(show: false),
          borderData: FlBorderData(show: false),
          lineBarsData: [
            // historic thick black line
            LineChartBarData(
              spots: _spots(hist, 0),
              isCurved: false,
              color: Colors.black,
              dotData: FlDotData(show: false),
              barWidth: 2.5,
            ),
            // median forecast
            LineChartBarData(
              spots: _spots(median, h),
              isCurved: false,
              color: Colors.orange,
              dotData: FlDotData(show: false),
              barWidth: 2,
            ),
            // p5 & p95 (make them invisible - we’ll fill the area instead)
            LineChartBarData(
              spots: _spots(p5, h),
              isCurved: false,
              color: Colors.transparent,
              belowBarData: BarAreaData(
                show: true,
                spotsLine: BarAreaSpotsLine(show: false),
                gradient: LinearGradient(
                  colors: [
                    Colors.orange.withOpacity(.15),
                    Colors.orange.withOpacity(.15),
                  ],
                ),
              ),
              dotData: FlDotData(show: false),
            ),
            LineChartBarData(
              spots: _spots(p95, h),
              isCurved: false,
              color: Colors.transparent,
              aboveBarData: BarAreaData(
                show: true,
                spotsLine: BarAreaSpotsLine(show: false),
                gradient: LinearGradient(
                  colors: [
                    Colors.orange.withOpacity(.15),
                    Colors.orange.withOpacity(.15),
                  ],
                ),
              ),
              dotData: FlDotData(show: false),
            ),
          ],
        ),
      ),
    );
  }
}
