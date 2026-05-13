import 'package:fitness_pose_app/main.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('home renders updated feature entries', (WidgetTester tester) async {
    await tester.pumpWidget(const FitnessPoseApp());

    expect(find.text('智能健身教练'), findsOneWidget);
    expect(find.text('深蹲训练'), findsOneWidget);
    expect(find.text('俯卧撑训练'), findsOneWidget);
    expect(find.text('平板支撑训练'), findsOneWidget);
    expect(find.text('身体标定'), findsOneWidget);
    expect(find.text('数据集采集'), findsOneWidget);
  });
}
