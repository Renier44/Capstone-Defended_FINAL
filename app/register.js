import { MaterialIcons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useState } from 'react';
import {
  Alert,
  KeyboardAvoidingView,
  Platform,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
  Image,
  ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import DateTimePicker from '@react-native-community/datetimepicker';
import * as SecureStore from 'expo-secure-store';
import { useFonts } from 'expo-font'; // <-- ADDED FONT IMPORT

const API_BASE_URL = 'https://capstone-defended-final.onrender.com';

export default function RegisterScreen() {
  const router = useRouter();
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [email, setEmail] = useState('');
  const [dateOfBirth, setDateOfBirth] = useState('');
  const [showDatePicker, setShowDatePicker] = useState(false);
  const [gender, setGender] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);

  const [loading, setLoading] = useState(false); // <-- ADD SPINNER STATE

  // Load required custom fonts for the brand name (SMART/SIGHT)
  const [fontsLoaded] = useFonts({
    SmartFont: require('../assets/fonts/ArchivoBlack-Regular.ttf'),
    SightFont: require('../assets/fonts/VarelaRound-Regular.ttf'),
  });

  if (!fontsLoaded) return null; // Wait for fonts to load


  const handleRegister = async () => {
    if (!firstName || !lastName || !email || !password || !dateOfBirth || !gender) {
      Alert.alert('Error', 'Please fill out all fields.');
      return;
    }

    if (password !== confirmPassword) {
      Alert.alert('Error', 'Passwords do not match.');
      return;
    }

    try {
      setLoading(true); // <-- START LOADING SPINNER

      const response = await fetch(`${API_BASE_URL}/api/register/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          first_name: firstName.trim(),
          last_name: lastName.trim(),
          email: email.trim(),
          date_of_birth: dateOfBirth.trim(),
          gender: gender.trim(),
          password: password.trim(),
        }),
      });

      const data = await response.json();
      console.log('Register response:', response.status, data);

      if (!response.ok) {
        setLoading(false);
        const errorMsg =
          data.errors
            ? Object.values(data.errors).flat().join('\n')
            : data.message || 'Something went wrong.';
        Alert.alert('Registration Failed', errorMsg);
        return;
      }

      if (data.token) await SecureStore.setItemAsync('userToken', data.token);
      if (data.user) await SecureStore.setItemAsync('userProfile', JSON.stringify(data.user));

      setLoading(false);
      Alert.alert('Success', 'Registration successful!');
      router.replace('/login');
    } catch (error) {
      setLoading(false);
      console.error('Register error:', error);
      Alert.alert('Error', 'Unable to register. Please check your network.');
    }
  };

  const onChangeDate = (event, selectedDate) => {
    if (Platform.OS === 'android') setShowDatePicker(false);
    if (selectedDate) {
      const formattedDate = selectedDate.toISOString().split('T')[0];
      setDateOfBirth(formattedDate);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      {/* Loading Overlay */}
      {loading && (
        <View style={styles.loadingOverlay}>
          <ActivityIndicator size="large" color="#0057B7" />
        </View>
      )}

      <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      >
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <MaterialIcons name="arrow-back-ios" size={24} color="#333" />
          <Text style={styles.backButtonText}>Back</Text>
        </TouchableOpacity>

        <View style={styles.content}>
          {/* Branding Row: STYLED AS PREVIOUS CODE (Stacked Text) */}
          <View style={styles.brandRow}>
            <View style={styles.brandTextContainer}>
              <Text style={styles.smartText}>SMART</Text>
              <Text style={styles.sightText}>SIGHT</Text>
            </View>
            <Image source={require('../assets/images/icon.png')} style={styles.logo} />
          </View>
          <Text style={styles.subBrand}>Enhance Vision Optical PH</Text>


          <View style={styles.inputGroup}>
            <Text style={styles.accessText}>Create your account below</Text>
            <View style={styles.rowContainer}>
              <View style={[styles.inputContainer, styles.halfInput]}>
                <TextInput
                  style={styles.input}
                  placeholder="First Name"
                  value={firstName}
                  onChangeText={setFirstName}
                  placeholderTextColor="#333"
                />
              </View>

              <View style={[styles.inputContainer, styles.halfInput]}>
                <TextInput
                  style={styles.input}
                  placeholder="Last Name"
                  value={lastName}
                  onChangeText={setLastName}
                  placeholderTextColor="#333"
                />
              </View>
            </View>

            <View style={styles.inputContainer}>
              <TextInput
                style={styles.input}
                placeholder="Email"
                keyboardType="email-address"
                autoCapitalize="none"
                value={email}
                onChangeText={setEmail}
                placeholderTextColor="#333"
              />
              <MaterialIcons name="mail-outline" size={24} color="#555" style={styles.icon} />
            </View>

            <View style={styles.inputContainer}>
              <TouchableOpacity style={styles.input} onPress={() => setShowDatePicker(true)}>
                <Text style={{ color: dateOfBirth ? '#000' : '#333' }}>
                  {dateOfBirth || 'Select Date of Birth'}
                </Text>
              </TouchableOpacity>
              <MaterialIcons name="calendar-today" size={24} color="#555" style={styles.icon} />

              {showDatePicker && (
                <DateTimePicker
                  value={dateOfBirth ? new Date(dateOfBirth) : new Date(2000, 0, 1)}
                  mode="date"
                  display="spinner"
                  onChange={onChangeDate}
                  maximumDate={new Date()}
                />
              )}
            </View>

            <View style={styles.genderContainer}>
              {['Male', 'Female'].map((option) => (
                <TouchableOpacity
                  key={option}
                  style={[
                    styles.genderOption,
                    gender === option.toLowerCase() && styles.genderOptionActive,
                  ]}
                  onPress={() => setGender(option.toLowerCase())}
                >
                  <Text
                    style={[
                      styles.genderText,
                      gender === option.toLowerCase() && styles.genderTextActive,
                    ]}
                  >
                    {option}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>

            <View style={styles.inputContainer}>
              <TextInput
                style={styles.input}
                placeholder="Password"
                secureTextEntry={!showPassword}
                value={password}
                onChangeText={setPassword}
                placeholderTextColor="#333"
              />
              <TouchableOpacity onPress={() => setShowPassword(!showPassword)} style={styles.icon}>
                <MaterialIcons name={showPassword ? 'lock-open' : 'lock'} size={24} color="#555" />
              </TouchableOpacity>
            </View>

            <View style={styles.inputContainer}>
              <TextInput
                style={styles.input}
                placeholder="Confirm Password"
                secureTextEntry={!showPassword}
                value={confirmPassword}
                onChangeText={setConfirmPassword}
                placeholderTextColor="#333"
              />
              <TouchableOpacity onPress={() => setShowPassword(!showPassword)} style={styles.icon}>
                <MaterialIcons name={showPassword ? 'lock-open' : 'lock'} size={24} color="#555" />
              </TouchableOpacity>
            </View>
          </View>

          

          {/* Register Button */}
          <TouchableOpacity
            style={[styles.nextButton, loading && { opacity: 0.5 }]}
            onPress={handleRegister}
            disabled={loading}
          >
            <Text style={styles.nextButtonText}>
              {loading ? 'Registering...' : 'Register'}
            </Text>
          </TouchableOpacity>

          <View style={styles.loginContainer}>
            <Text style={styles.alreadyMemberText}>Already a member?</Text>
            <TouchableOpacity onPress={() => router.push('/login')}>
              <Text style={styles.loginNowText}>Login now</Text>
            </TouchableOpacity>
          </View>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff' },

  // Spinner overlay (Kept from current code)
  loadingOverlay: {
    position: 'absolute',
    zIndex: 999,
    backgroundColor: 'rgba(0,0,0,0.2)',
    top: 0,
    left: 0,
    height: '100%',
    width: '100%',
    alignItems: 'center',
    justifyContent: 'center',
  },

  backButton: { flexDirection: 'row', alignItems: 'center', padding: 20, marginTop: 25, },
  backButtonText: { marginLeft: 5, fontSize: 16, color: '#333' },

  // Updated style to center content vertically (from previous code)
  content: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center', // Transferred style
    paddingHorizontal: 20,
    
  },

  // Updated style for centralized branding (from previous code)
  brandRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center', // Transferred style
    marginBottom: 15, // Transferred style
    marginTop: -70,
  },

  // New container needed for stacked logo text (from previous code)
  brandTextContainer: {
    marginRight: 12,
    alignItems: 'flex-end',
  },

  // Branding text style (Transferred style - simplified font family)
  smartText: {
    fontSize: 40,
    color: '#77CDE0',
    fontFamily: 'SmartFont', // <-- APPLIED FONT
    lineHeight: 42,
  },

  // Branding text style (Transferred style - simplified font family)
  sightText: {
    fontSize: 40,
    color: '#FFD54F',
    fontFamily: 'SightFont', // <-- APPLIED FONT
    lineHeight: 42,
    marginRight: 5,
    marginBottom: -15,
    
  },

  logo: { width: 90, height: 90, resizeMode: 'contain', marginTop: 25, }, // Transferred size

  // Sub brand text (Transferred style)
  subBrand: { fontSize: 15, color: '#333', marginBottom: 20 },

  // Input Group container style (Transferred background color from previous code)
  inputGroup: {
    width: '100%',
    maxWidth: 320,
    backgroundColor: '#99E0E9', // Transferred style
    borderRadius: 20,
    padding: 25,
    marginBottom: 150,

  },

  // Text inside input group (Transferred style)
  accessText: {
    fontSize: 14,
    color: '#555',
    textAlign: 'center',
    marginBottom: 20,
  },

  // Styles for the new row layout (Kept from current code)
  rowContainer: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 15 },
  halfInput: { flex: 1, marginHorizontal: 5 },

  // Input container style (Kept from current code)
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#C8EAF7',
    borderRadius: 12,
    paddingHorizontal: 15,
    marginBottom: 15,
    height: 50,
  },
  input: { flex: 1, fontSize: 16, color: '#000' },
  icon: { marginLeft: 10 ,},
  

  // Gender selection styles (Kept from current code)
  genderContainer: {
    flexDirection: 'row',
    gap: 10,
    justifyContent: 'space-between',
    marginBottom: 15,
  },
  genderOption: {
    flex: 1,
    backgroundColor: '#C8EAF7',
    borderRadius: 12,
    height: 50,
    alignItems: 'center',
    justifyContent: 'center',
  },
  genderOptionActive: { backgroundColor: '#4C6CD5' },
  genderText: { color: '#333', fontSize: 16 },
  genderTextActive: { color: '#fff', fontWeight: '600' },

  showPasswordText: {
    alignSelf: 'flex-end',
    width: '100%',
    maxWidth: 320,
    textAlign: 'right',
    color: '#5B4FE9',
    fontSize: 13,
    marginBottom: 20,
    marginTop: -130,
  },

  nextButton: {
    backgroundColor: '#FFD54F',
    paddingVertical: 15,
    paddingHorizontal: 60,
    borderRadius: 30,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: -130,
    elevation: 3,
  },
  nextButtonText: { color: '#fff', fontSize: 18, fontWeight: 'bold' },

  loginContainer: { flexDirection: 'row', marginTop: 30 },
  alreadyMemberText: { fontSize: 15, color: '#333', marginTop: -20, },
  loginNowText: { fontSize: 15, color: '#5B4FE9', fontWeight: '600', marginLeft: 5 , marginTop: -20,},
});