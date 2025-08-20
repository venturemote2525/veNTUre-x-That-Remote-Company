import Header from '@/components/Header';
import { ThemedSafeAreaView } from '@/components/Themed';
import { useLocalSearchParams } from 'expo-router';
import { Image } from 'react-native';

export default function ConfirmScreen() {
  const { image } = useLocalSearchParams<{ image: string }>();
  if (!image) return null;
  return (
    <ThemedSafeAreaView>
      <Header title="Confirm Image" />
      <Image
        source={{ uri: decodeURIComponent(image) }}
        style={{ width: 300, height: 300, alignSelf: 'center', marginTop: 20 }}
      />
    </ThemedSafeAreaView>
  );
}
