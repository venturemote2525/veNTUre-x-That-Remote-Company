import { Text, ThemedSafeAreaView, View } from '@/components/Themed';

export default function LoggingScreen() {
  return (
    <ThemedSafeAreaView>
      <View className="flex-1 items-center justify-center">
        <Text>Logging</Text>
      </View>
    </ThemedSafeAreaView>
  );
}
