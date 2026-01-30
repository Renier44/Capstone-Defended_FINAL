import { Stack } from 'expo-router';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { ImageProvider } from './context/ImageContext';

export default function Layout() {
  return (
    <SafeAreaProvider>
      <ImageProvider>
        <Stack screenOptions={{ headerShown: false }} />
      </ImageProvider>
    </SafeAreaProvider>
  );
}
