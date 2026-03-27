import { Ionicons } from '@expo/vector-icons';
import DateTimePickerModal from 'react-native-modal-datetime-picker';
import { useRouter, useLocalSearchParams } from 'expo-router';
import { useState, useEffect } from 'react';
import * as SecureStore from 'expo-secure-store';
import {
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
  ActivityIndicator,
  Alert,
} from 'react-native';


export default function BookAppointment() {
  const router = useRouter();
  const { reason: aiReason, aiScreening, isEdit, ...editParams } = useLocalSearchParams();

  const [bookingFor, setBookingFor] = useState('');
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [email, setEmail] = useState('');
  const [dob, setDob] = useState('');
  const [showDatePicker, setShowDatePicker] = useState(false);
  const [gender, setGender] = useState('');
  const [age, setAge] = useState('');
  const [reason, setReason] = useState('');

  const [reasonsList, setReasonsList] = useState([]);
  const [loading, setLoading] = useState(true);
  const [genderChoices, setGenderChoices] = useState([]);
  const [loadingGenders, setLoadingGenders] = useState(true);
  const [bookingForChoices, setBookingForChoices] = useState([]);
  const [loadingBookingFor, setLoadingBookingFor] = useState(true);

  const API_BASE_URL = 'https://capstone-defended-final.onrender.com';
  const [userProfile, setUserProfile] = useState(null);

  // Load user profile from SecureStore
  const loadUserProfile = async () => {
    try {
      const profileStr = await SecureStore.getItemAsync('userProfile');
      if (profileStr) setUserProfile(JSON.parse(profileStr));
    } catch (err) {
      console.error('Error loading user profile:', err);
    }
  };

  useEffect(() => { loadUserProfile(); }, []);

  // Set values if editing appointment
  useEffect(() => {
    if (isEdit) {
      setFirstName(editParams.firstName || '');
      setLastName(editParams.lastName || '');
      setEmail(editParams.email || '');
      setDob(editParams.dob || '');
      setGender(editParams.gender || '');
      setAge(editParams.age ? editParams.age.toString() : '');
      setReason(editParams.reason || '');
      setBookingFor(editParams.bookingFor || '');
    }
  }, [isEdit]);

  // Fetch reasons list or set AI reason
  useEffect(() => {
    const fetchReasons = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/appointment/reasons/`);
        const data = await response.json();
        setReasonsList(data);
      } catch (error) {
        console.error('Error fetching reasons:', error);
        Alert.alert('Error', 'Unable to load reasons.');
      } finally { setLoading(false); }
    };

    if (aiScreening && aiReason) {
      const formattedReason = aiReason
        .replace(/(Diagnosis:)/, '$1 ')
        .replace(/(Confidence:)/, '\n$1 ');
      setReason(formattedReason.trim());
      setLoading(false);
    } else fetchReasons();
  }, [aiScreening, aiReason]);

  // Fetch gender choices
  useEffect(() => {
    const fetchGenderChoices = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/appointment/genders/`);
        const data = await response.json();
        setGenderChoices(data);
        if (!isEdit && data.length > 0) setGender(data[0]);
      } catch (error) { console.error('Error fetching genders:', error); }
      finally { setLoadingGenders(false); }
    };
    fetchGenderChoices();
  }, []);

  // Fetch bookingFor choices
  useEffect(() => {
    const fetchBookingForChoices = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/appointment/booking-for/`);
        const data = await response.json();
        setBookingForChoices(data);
        if (!isEdit && data.length > 0) setBookingFor(data[0]);
      } catch (error) { console.error('Error fetching booking for:', error); }
      finally { setLoadingBookingFor(false); }
    };
    fetchBookingForChoices();
  }, []);

  // Auto-fill user info if booking for yourself
  useEffect(() => {
  if (!userProfile) return;

  const isYourself = bookingFor.toLowerCase() === 'yourself';
  const isAnother = bookingFor.toLowerCase() === 'another';

  if (isYourself) {
    // Autofill all fields from userProfile
    setFirstName(userProfile.first_name || '');
    setLastName(userProfile.last_name || '');
    setEmail(userProfile.email || '');
    setGender(userProfile.gender || '');
    if (userProfile.date_of_birth) {
      const dobDate = new Date(userProfile.date_of_birth);
      const formattedDate = `${dobDate.getFullYear()}-${String(dobDate.getMonth() + 1).padStart(2,'0')}-${String(dobDate.getDate()).padStart(2,'0')}`;
      setDob(formattedDate);
      setAge(calculateAge(dobDate).toString());
    } else {
      setDob('');
      setAge('');
    }
  }

  if (isAnother) {
    setFirstName('');
    setLastName('');
    setEmail('');
    setGender('');
    setDob('');
    setAge('');
  }
}, [bookingFor, userProfile]);


  const calculateAge = (birthDate) => {
    const today = new Date();
    let years = today.getFullYear() - birthDate.getFullYear();
    const m = today.getMonth() - birthDate.getMonth();
    if (m < 0 || (m === 0 && today.getDate() < birthDate.getDate())) years--;
    return years;
  };

  const handleNext = async () => {
    if (!firstName || !lastName || !dob || !age || !reason || !gender || !email) {
      Alert.alert('Missing Details', 'Please complete all required fields.');
      return;
    }

    if (isEdit) {
      try {
        const token = await SecureStore.getItemAsync('userToken');
        const response = await fetch(`${API_BASE_URL}/api/edit-appointment/${editParams.id}/`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json', Authorization: `Token ${token}` },
          body: JSON.stringify({
            first_name: firstName,
            last_name: lastName,
            email: email,
            date_of_birth: dob,
            gender: gender,
            age: parseInt(age),
            reason: reason,
            booking_for: bookingFor,
          }),
        });

        const text = await response.text();
        console.log('Update response:', text);

        if (response.ok) {
          Alert.alert('Success', 'Appointment updated successfully.', [
            { text: 'OK', onPress: () => router.replace('/my-appointments') }
          ]);
        } else {
          Alert.alert('Error', 'Failed to update appointment.');
        }
      } catch (error) {
        console.error('Error updating appointment:', error);
        Alert.alert('Error', 'Something went wrong.');
      }
      return;
    }

    router.push({
      pathname: '/schedule',
      params: {
        firstName,
        lastName,
        email,
        dob,
        gender,
        age: parseInt(age),
        reason,
        bookingFor
      }
    });
  };

  const formatText = (text) => text.charAt(0).toUpperCase() + text.slice(1).replace('-', ' ');

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()}>
          <Ionicons name="arrow-back" size={26} color="#fff" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>{isEdit ? 'Edit Appointment' : 'Patient Details'}</Text>
      </View>

      <View style={styles.content}>
        {/* Booking For */}
        <Text style={styles.label}>Booking For</Text>
        {loadingBookingFor ? <ActivityIndicator size="small" color="#fff" /> :
          <View style={styles.toggleWrapper}>
            {bookingForChoices.map((option) => (
              <TouchableOpacity
                key={option}
                style={[styles.toggleOption, bookingFor === option && styles.toggleOptionActive]}
                onPress={() => setBookingFor(option)}
              >
                <Text style={[styles.toggleText, bookingFor === option && styles.toggleTextActive]}>{formatText(option)}</Text>
              </TouchableOpacity>
            ))}
          </View>
        }

        {/* Name Inputs */}
        <View style={styles.inputRow}>
          <TextInput style={[styles.inputField, { flex: 1 }]} placeholder="First Name" value={firstName} onChangeText={setFirstName} editable={bookingFor.toLowerCase() === 'another'} placeholderTextColor="#555" />
          <TextInput style={[styles.inputField, { flex: 1, marginLeft: 10 }]} placeholder="Last Name" value={lastName} onChangeText={setLastName} editable={bookingFor.toLowerCase() === 'another'} placeholderTextColor="#555" />
        </View>

        {/* Email */}
        <Text style={styles.label}>Email</Text>
        <TextInput style={[styles.inputSmall, { width: '48%' }]} placeholder="Email Address" value={email} onChangeText={setEmail} editable={bookingFor.toLowerCase() === 'another'} keyboardType="email-address" placeholderTextColor="#555" />

        {/* DOB */}
        <Text style={styles.label}>Date of Birth</Text>
        <TouchableOpacity style={[styles.inputSmall, { width: '48%' }]} onPress={() => setShowDatePicker(true)}>
          <Text style={{ color: dob ? '#000' : '#555' }}>{dob || 'Select date'}</Text>
        </TouchableOpacity>
        <DateTimePickerModal
          isVisible={showDatePicker}
          mode="date"
          maximumDate={new Date()}
          onConfirm={(date) => {
            setShowDatePicker(false);
            const formattedDate = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2,'0')}-${String(date.getDate()).padStart(2,'0')}`;
            setDob(formattedDate);
            setAge(calculateAge(date).toString());
          }}
          onCancel={() => setShowDatePicker(false)}
        />

        {/* Gender */}
        <Text style={styles.label}>Gender</Text>
        {loadingGenders ? <ActivityIndicator size="small" color="#fff" /> :
          <View style={styles.genderWrapper}>
            {genderChoices.map((option, idx) => (
              <TouchableOpacity
                key={option}
                style={[styles.genderOption, gender === option && styles.genderOptionActive, idx !== 0 && { marginLeft: 10 }]}
                onPress={() => setGender(option)}
              >
                <Text style={[styles.genderText, gender === option && styles.genderTextActive]}>{option}</Text>
              </TouchableOpacity>
            ))}
          </View>
        }

        {/* Age */}
        <Text style={styles.label}>Age</Text>
        <TextInput style={[styles.inputSmall, { width: '48%' }]} placeholder="Age" value={age} editable={false} />

        {/* Reason */}
        <Text style={styles.label}>{aiScreening ? 'Preliminary Result' : 'Reason for Appointment'}</Text>
        {loading ? <ActivityIndicator size="large" color="#fff" /> : aiScreening ? (
          <TextInput style={[styles.reasonInput, { backgroundColor: '#C8EAF7', lineHeight: 22 }]} value={reason} editable={false} multiline />
        ) : (
          <View style={styles.reasonContainer}>
            {reasonsList.map((item, idx) => (
              <TouchableOpacity key={idx} style={[styles.reasonItem, reason === item && styles.reasonItemSelected]} onPress={() => setReason(item)}>
                <Text style={[styles.reasonText, reason === item && styles.reasonTextSelected]}>{item}</Text>
              </TouchableOpacity>
            ))}
          </View>
        )}

        <TouchableOpacity style={styles.nextButton} onPress={handleNext}>
          <Text style={styles.nextButtonText}>{isEdit ? 'Update Appointment' : 'Next'}</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#FFFF0' },
  header: { flexDirection: 'row', alignItems: 'center', paddingTop: 60, paddingHorizontal: 20, paddingBottom: 20 },
  headerTitle: { fontSize: 24, fontWeight: 'bold', fontFamily: 'League Spartan', color: '#2260FF', textAlign: 'center', width: '100%' },
  content: { paddingHorizontal: 20, paddingBottom: 40 },
  label: { fontSize: 15, fontFamily: 'League Spartan', marginBottom: 5, color: '#000' },

  toggleWrapper: { flexDirection: 'row', backgroundColor: 'rgba(255,255,255,0.5)', borderRadius: 30, marginBottom: 15, padding: 4 },
  toggleOption: { flex: 1, paddingVertical: 10, alignItems: 'center', borderRadius: 25 },
  toggleOptionActive: { backgroundColor: '#FFD54F' },
  toggleText: { fontSize: 14, color: '#000' },
  toggleTextActive: { color: '#000', fontWeight: 'bold' },

  inputRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 12 },
  inputField: { borderWidth: 1, borderColor: '#bdc3c7', paddingHorizontal: 10, borderRadius: 20, fontSize: 14, backgroundColor: '#C8EAF7', height: 36, color: '#000' },
  inputSmall: { borderWidth: 1, borderColor: '#bdc3c7', paddingHorizontal: 8, borderRadius: 20, fontSize: 14, backgroundColor: '#C8EAF7', height: 36, color: '#000' },

  genderWrapper: { flexDirection: 'row', backgroundColor: '#C8EAF7', borderRadius: 50, padding: 2, marginBottom: 12 },
  genderOption: { flex: 1, paddingVertical: 8, alignItems: 'center', borderRadius: 50 },
  genderOptionActive: { backgroundColor: '#FFD54F' },
  genderText: { fontSize: 14, color: '#000' },
  genderTextActive: { fontWeight: 'bold' },
  
  reasonContainer: { backgroundColor: '#C8EAF7', padding: 8, borderRadius: 30, marginBottom: 15 },
  reasonItem: { padding: 8, borderRadius: 30, marginBottom: 8, borderWidth: 1, borderColor: '#bdc3c7', alignItems: 'center' },
  reasonItemSelected: { backgroundColor: '#FFD54F', borderColor: '#FFD54F' },
  reasonText: { fontSize: 14, color: '#000' },
  reasonTextSelected: { color: '#000', fontWeight: 'bold' },
  reasonInput: { borderWidth: 1, borderColor: '#bdc3c7', paddingVertical: 10, paddingHorizontal: 15, borderRadius: 30, fontSize: 14, marginBottom: 15, backgroundColor: '#C8EAF7', textAlign: 'center', fontWeight: 'bold', color: '#000' },

  nextButton: { backgroundColor: '#FFD54F', padding: 14, borderRadius: 30, alignItems: 'center', marginTop: 20, elevation: 5 },
  nextButtonText: { color: '#000', fontSize: 16, fontWeight: 'bold', fontFamily: 'League Spartan' },
});
