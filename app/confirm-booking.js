import { Ionicons } from '@expo/vector-icons';
import { useLocalSearchParams, useRouter } from 'expo-router';
import {
  Image,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  ActivityIndicator,
} from 'react-native';

import { useState } from 'react';
import * as SecureStore from 'expo-secure-store';

export default function ConfirmBooking() {
  const router = useRouter();
  const params = useLocalSearchParams();

  const {
  firstName = '',
  lastName = '',
  email = '',
  gender = '',
  age = '',
  reason = '',
  bookingFor = '',
  selectedTime = '',
  appointmentDate = '',
  doctorId = '',
  assignedDoctorName = '',
  doctorImage = null,   // ✅ Add this line
  is_ai_screening = 'false',
  preliminary_result = '',
} = params;

  const [submitting, setSubmitting] = useState(false);

  const formatDateTime = (date, time) => `${date} ${time}`;

  const handleConfirm = async () => {
    setSubmitting(true);
    try {
      const full_datetime_str = formatDateTime(appointmentDate, selectedTime);
      const token = await SecureStore.getItemAsync('userToken');

      // ✅ Explicit detection — either passed in params or has preliminary result
      const isAIScreening =
        is_ai_screening === 'true' ||
        preliminary_result !== '' ||
        reason?.toLowerCase().includes('diagnosis');

      // ✅ Set correct field logic
      const appointmentPayload = {
        firstName,
        lastName,
        age: parseInt(age),
        gender,
        bookingFor,
        patient_email: email,
        appointment_datetime_str: full_datetime_str,
        doctorId: parseInt(doctorId, 10),
        is_ai_screening: isAIScreening,
        preliminary_result: isAIScreening
          ? preliminary_result || reason || 'Diagnosis: Pending AI result'
          : null,
        reason: isAIScreening ? null : reason,
      };

      const res = await fetch(
        'https://capstone-defended-final.onrender.com/api/appointments/create/',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(token ? { Authorization: `Token ${token}` } : {}),
          },
          body: JSON.stringify(appointmentPayload),
        }
      );

      const data = await res.json();

      if (!res.ok) {
        router.push({
          pathname: '/confirmation',
          params: {
            success: 'false',
            message:
              data.error ||
              'The selected timeslot is no longer available. Please choose another one.',
          },
        });
        return;
      }

      router.push({
        pathname: '/confirmation',
        params: {
          success: 'true',
          message: isAIScreening
            ? 'Your AI Screening appointment has been submitted successfully.'
            : 'Your appointment has been submitted successfully.',
        },
      });
    } catch (err) {
      router.push({
        pathname: '/confirmation',
        params: {
          success: 'false',
          message: 'Network error. Could not reach the server.',
        },
      });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={26} color="#2260FF" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Appointment Review</Text>
      </View>

      {/* Doctor Card */}
<View style={styles.card}>
  <View style={styles.doctorRow}>
    <Image
      source={
        doctorImage // use the local asset passed from ScheduleScreen
          ? doctorImage
          : {
              uri: `https://ui-avatars.com/api/?name=${assignedDoctorName}&background=2260FF&color=fff&size=100&bold=true`,
            }
      }
      style={styles.doctorImage}
    />
    <View style={styles.doctorInfo}>
      <Text style={styles.doctorName}>{assignedDoctorName}</Text>
      <Text style={styles.doctorTitle}>Optometrist</Text>
    </View>
  </View>
</View>


      {/* Appointment Info */}
      <View style={styles.card}>
        <Text style={styles.sectionHeader}>📅 Appointment Details</Text>
        <View style={styles.detailRow}>
          <Text style={styles.label}>Date</Text>
          <Text style={styles.value}>{appointmentDate || 'Not set'}</Text>
        </View>
        <View style={styles.detailRow}>
          <Text style={styles.label}>Time</Text>
          <Text style={styles.value}>{selectedTime || 'Not set'}</Text>
        </View>
      </View>

      {/* Patient Info */}
      <View style={styles.card}>
        <Text style={styles.sectionHeader}>👤 Patient Information</Text>

        <View style={styles.detailRow}>
          <Text style={styles.label}>Booking For</Text>
          <Text style={styles.value}>
            {bookingFor === 'yourself' ? 'Yourself' : bookingFor || 'N/A'}
          </Text>
        </View>

        <View style={styles.detailRow}>
          <Text style={styles.label}>Last Name</Text>
          <Text style={styles.value}>{lastName || 'N/A'}</Text>
        </View>

        <View style={styles.detailRow}>
          <Text style={styles.label}>First Name</Text>
          <Text style={styles.value}>{firstName || 'N/A'}</Text>
        </View>

        <View style={styles.detailRow}>
          <Text style={styles.label}>Email</Text>
          <Text style={styles.value}>{email || 'N/A'}</Text>
        </View>

        <View style={styles.detailRow}>
          <Text style={styles.label}>Age</Text>
          <Text style={styles.value}>{age || 'N/A'}</Text>
        </View>

        <View style={styles.detailRow}>
          <Text style={styles.label}>Gender</Text>
          <Text style={styles.value}>{gender || 'N/A'}</Text>
        </View>

        {/* ✅ Label dynamically changes */}
        <View style={styles.detailRow}>
          <Text style={styles.label}>
            {is_ai_screening === 'true' || reason?.toLowerCase().includes('diagnosis')
              ? 'Preliminary Result'
              : 'Reason'}
          </Text>
          <Text style={styles.value}>{reason || preliminary_result || 'N/A'}</Text>
        </View>
      </View>

      {/* Confirm Button */}
      <TouchableOpacity
        style={[styles.confirmButton, submitting && { opacity: 0.7 }]}
        onPress={handleConfirm}
        disabled={submitting}
      >
        {submitting ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.confirmButtonText}>Confirm Booking</Text>
        )}
      </TouchableOpacity>
    </ScrollView>
  );
}

// your styles here...


const styles = StyleSheet.create({
  // Background and Layout
  container: {
    flex: 1,
    backgroundColor: '#C8EAF7', 
    paddingHorizontal: 20, 
  },
  content: {
    paddingTop: 0, 
    paddingBottom: 30, 
    paddingHorizontal: 0,
  },

  // --- Header ---
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingTop: 50, 
    marginBottom: 20, 
  },
  backButton: {
    paddingRight: 15, 
  },
  headerTitle: {
    flex: 1,
    textAlign: 'center',
    fontSize: 22,
    fontWeight: 'bold', 
    color: '#2260FF',
    fontFamily: 'LeagueSpartan', 
    marginRight: 26,
  },

  // --- Reusable Card ---
  card: {
    backgroundColor: '#FFFFFF',
    borderRadius: 25,
    padding: 20,
    marginVertical: 10,
    shadowColor: '#000',
    shadowOpacity: 0.1,
    shadowOffset: { width: 0, height: 4 },
    shadowRadius: 5,
    elevation: 8,
    // FIX: Add side borders
    borderLeftWidth: 4, 
    borderRightWidth: 4,
    borderColor: '#77CDE0', 
  },

  // --- Doctor Section ---
  doctorRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  doctorImage: {
    width: 80, 
    height: 80,
    borderRadius: 40,
    marginRight: 15,
  },
  doctorInfo: {
    flex: 1,
  },
  doctorName: {
    fontSize: 20, 
    fontWeight: 'bold',
    color: '#2260FF',
    fontFamily: 'LeagueSpartan',
  },
  doctorTitle: {
    fontSize: 15,
    color: '#555',
    marginTop: 2,
    fontFamily: 'LeagueSpartan',
  },

  // --- Details and Text Layout ---
  sectionHeader: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2260FF',
    fontFamily: 'LeagueSpartan',
    marginBottom: 10,
    borderBottomWidth: 0, 
    paddingBottom: 0,
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 10, 
    borderBottomWidth: 0, 
    paddingBottom: 0,
  },
  label: {
    fontWeight: '600',
    color: '#555', 
    fontSize: 14,
    fontFamily: 'LeagueSpartan',
  },
  value: {
    fontSize: 14,
    color: '#333',
    textAlign: 'right',
    flexShrink: 1,
    fontFamily: 'LeagueSpartan',
  },

  // --- Confirm Button ---
  confirmButton: {
    backgroundColor: '#FFD54F', 
    paddingVertical: 13,
    borderRadius: 25,
    alignItems: 'center',
    marginTop: 30,
    marginBottom: 30, 
    shadowColor: '#000',
    shadowOpacity: 0.2,
    shadowOffset: { width: 0, height: 4 },
    shadowRadius: 6,
    elevation: 8,
  },
  confirmButtonText: {
    color: '#fff', 
    textAlign: 'center',
    fontWeight: 'bold',
    fontSize: 18,
    fontFamily: 'LeagueSpartan',
    letterSpacing: 0,
  },
});