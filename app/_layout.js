import { Stack } from 'expo-router';
import { SafeAreaProvider } from 'react-native-safe-area-context';

import { UserProvider } from '../context/UserContext';
import { ImageProvider } from '../context/ImageContext';

export default function Layout() {
  return (
    <SafeAreaProvider>
      <UserProvider>
        <ImageProvider>
          <Stack screenOptions={{ headerShown: false }} />
        </ImageProvider>
      </UserProvider>
    </SafeAreaProvider>
  );
}
