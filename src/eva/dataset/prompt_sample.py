import random


def prompt_overview(items):
    prompt = []
    if items['Modality']=='magnetic resonance imaging' and items['Sequence']=='diffusion weighted imaging':
        prompt.append({
            'content': [
                'A {} with b-value of {} of the {} for the subject.'.format('{} {}'.format(items['Sequence'], items['Modality']) if items['Modality']=='magnetic resonance imaging' else items['Modality'], items['bvalue'], items['Organ'][0]),
                'A {} with b-value of {} for the subject.'.format('{} {}'.format(items['Sequence'], items['Modality']) if items['Modality']=='magnetic resonance imaging' else items['Modality'], items['bvalue']),
            ],
            'mask': 1, # 1 for necessary, 0 for random, -1 for omit
        })
    else:
        prompt.append({
            'content': [
                'A {} of the {} for the subject.'.format('{} {}'.format(items['Sequence'], items['Modality']) if items['Modality']=='magnetic resonance imaging' else items['Modality'], items['Organ'][0]),
                'A {} for the subject.'.format('{} {}'.format(items['Sequence'], items['Modality']) if items['Modality']=='magnetic resonance imaging' else items['Modality']),
            ],
            'mask': 1, # 1 for necessary, 0 for random, -1 for omit
        })
    for o in items['Organ'][1:]:
        prompt[-1]['content'].append('A {} of the {} for the subject.'.format('{} {}'.format(items['Sequence'], items['Modality']) if items['Modality']=='magnetic resonance imaging' else items['Modality'], o))

    if 'Contrast' in items:
        if items['Contrast']=='unknown':
            content = [' The subject undergoes the contrast-enhanced scan.']
        else:
            content = [
                ' The subject undergoes the contrast-enhanced scan with the {} agent.'.format(items['Contrast']),
                ' The subject undergoes the contrast-enhanced scan.',
            ]
        prompt.append({
            'content': content,
            'mask': 1, # 1 for necessary, 0 for random, -1 for omit
        })

    if 'ManufacturerModelName' in items and 'MagneticFieldStrength' in items:
        prompt.append({
            'content': [
                ' The scan is acquired on the {} {:.1f} Tesla scanner.'.format(items['ManufacturerModelName'], items['MagneticFieldStrength']),
                ' The scan is acquired on the {} scanner.'.format(items['ManufacturerModelName']),
                ' The scan is acquired on the {:.1f} Tesla scanner.'.format(items['MagneticFieldStrength']),
            ],
            'mask': 0, # 1 for necessary, 0 for random, -1 for omit
        })
    elif 'ManufacturerModelName' in items:
        prompt.append({
            'content': [
                ' The scan is acquired on the {} scanner.'.format(items['ManufacturerModelName']),
            ],
            'mask': 0, # 1 for necessary, 0 for random, -1 for omit
        })
    elif 'MagneticFieldStrength' in items:
        prompt.append({
            'content': [
                ' The scan is acquired on the {:.1f} Tesla scanner.'.format(items['MagneticFieldStrength']),
            ],
            'mask': 0, # 1 for necessary, 0 for random, -1 for omit
        })

    return prompt


def prompt_ct(items):
    prompt = []

    if 'KVP' in items and 'ExposureTime' in items and 'XRayTubeCurrent' in items:
        prompt.append({
            'content': [
                ' The scan is performed at the tube voltage of {:.1f} kVp and the tube current of {:.1f} mA, with an exposure time of {:.1f} ms.'.format(items['KVP'], items['XRayTubeCurrent'], items['ExposureTime']),
                ' The scan is performed at the tube voltage of {:.1f} kVp and the tube current of {:.1f} mA.'.format(items['KVP'], items['XRayTubeCurrent']),
                ' The scan is performed at the tube voltage of {:.1f} kVp, with an exposure time of {:.1f} ms.'.format(items['KVP'], items['ExposureTime']),
                ' The scan is performed at the tube voltage of {:.1f} kVp.'.format(items['KVP']),
            ],
            'mask': 0, # 1 for necessary, 0 for random, -1 for omit
        })
    elif 'KVP' in items and 'XRayTubeCurrent' in items:
        prompt.append({
            'content': [
                ' The scan is performed at the tube voltage of {:.1f} kVp and the tube current of {:.1f} mA.'.format(items['KVP'], items['XRayTubeCurrent']),
                ' The scan is performed at the tube voltage of {:.1f} kVp.'.format(items['KVP']),
            ],
            'mask': 0, # 1 for necessary, 0 for random, -1 for omit
        })
    elif 'KVP' in items and 'ExposureTime' in items:
        prompt += ' The scan is performed at the tube voltage of {:.1f} kVp, with an exposure time of {:.1f} ms.'.format(
            items['KVP'], items['ExposureTime'],
        )
        prompt.append({
            'content': [
                ' The scan is performed at the tube voltage of {:.1f} kVp, with an exposure time of {:.1f} ms.'.format(items['KVP'], items['ExposureTime']),
                ' The scan is performed at the tube voltage of {:.1f} kVp.'.format(items['KVP']),
            ],
            'mask': 0, # 1 for necessary, 0 for random, -1 for omit
        })
    elif 'KVP' in items:
        prompt.append({
            'content': [
                ' The scan is performed at the tube voltage of {:.1f} kVp.'.format(items['KVP']),
            ],
            'mask': 0, # 1 for necessary, 0 for random, -1 for omit
        })
        
    return prompt


def prompt_mr(items):
    prompt = []

    if items['Sequence']=='dynamic contrast-enhanced':
        if items['SequenceFrameNumber']==0:
            content = [' This is the pre-contrast scan before contrast injection.']
        else:
            number2ordinal = {
                1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', 6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth',
                11: '11th', 12: '12th', 13: '13th', 14: '14th', 15: '15th', 16: '16th', 17: '17th', 18: '18th', 19: '19th', 20: '20th',
                21: '21st', 22: '22nd', 23: '23th', 24: '24th', 25: '25th', 26: '26th', 27: '27th', 28: '28th', 29: '29th', 30: '30th',
                31: '31st', 32: '32nd', 33: '33th', 34: '34th', 35: '35th', 36: '36th', 37: '37th', 38: '38th', 39: '39th', 40: '40th',
                41: '41st', 42: '42nd', 43: '43th', 44: '44th', 45: '45th', 46: '46th', 47: '47th', 48: '48th', 49: '49th', 50: '50th',
            }
            if 'AcquisitionTime' not in items:
                content = [' This is the {} post-contrast scan after contrast injection.'.format(number2ordinal[items['SequenceFrameNumber']])]
            else:
                content = [
                    ' This is the {} post-contrast scan at {:.1f} s after contrast injection.'.format(number2ordinal[items['SequenceFrameNumber']], items['AcquisitionTime']),
                    ' This is the {} post-contrast scan after contrast injection.'.format(number2ordinal[items['SequenceFrameNumber']]),
                ]
        prompt.append({
            'content': content,
            'mask': 1, # 1 for necessary, 0 for random, -1 for omit
        })
    
    if 'Coil' in items:
        prompt.append({
            'content': [' The scan is acquired using a {}.'.format(items['Coil'])],
            'mask': 0, # 1 for necessary, 0 for random, -1 for omit
        })
    if 'ScanningSequence' in items:
        prompt.append({
            'content': [' The scanning sequences consist of {}.'.format(items['ScanningSequence'])],
            'mask': 1 if 'fat saturation' in items['ScanningSequence'] else 0, # 1 for necessary, 0 for random, -1 for omit
        })
    if 'SequenceVariant' in items:
        prompt.append({
            'content': [' The sequence variants consist of {}.'.format(items['SequenceVariant'])],
            'mask': 1 if 'fat saturation' in items['SequenceVariant'] else 0, # 1 for necessary, 0 for random, -1 for omit
        })
    if 'ScanOptions' in items:
        prompt.append({
            'content': [' The scan options consist of {}.'.format(items['ScanOptions'])],
            'mask': 1 if 'fat saturation' in items['ScanOptions'] else 0, # 1 for necessary, 0 for random, -1 for omit
        })
    
    if 'EchoTime' in items:
        if 'FlipAngle' in items and 'InversionTime' in items:
            content = [
                ' The image is acquired using a {:.1f} degree flip angle, an echo time of {:.1f} ms, a repetition time of {:.1f} ms, an inversion time of {:.1f} ms.'.format(items['FlipAngle'], items['EchoTime'], items['RepetitionTime'], items['InversionTime']),
                ' The image is acquired using an echo time of {:.1f} ms, a repetition time of {:.1f} ms, an inversion time of {:.1f} ms.'.format(items['EchoTime'], items['RepetitionTime'], items['InversionTime']),
                ' The image is acquired using a {:.1f} degree flip angle, an echo time of {:.1f} ms, a repetition time of {:.1f} ms.'.format(items['FlipAngle'], items['EchoTime'], items['RepetitionTime']),
                ' The image is acquired using an echo time of {:.1f} ms, a repetition time of {:.1f} ms.'.format(items['EchoTime'], items['RepetitionTime']),
            ]
        elif 'FlipAngle' in items:
            content = [
                ' The image is acquired using a {:.1f} degree flip angle, an echo time of {:.1f} ms, a repetition time of {:.1f} ms.'.format(items['FlipAngle'], items['EchoTime'], items['RepetitionTime']),
                ' The image is acquired using an echo time of {:.1f} ms, a repetition time of {:.1f} ms.'.format(items['EchoTime'], items['RepetitionTime']),
            ]
        elif 'InversionTime' in items:
            content = [
                ' The image is acquired using an echo time of {:.1f} ms, a repetition time of {:.1f} ms, an inversion time of {:.1f} ms.'.format(items['EchoTime'], items['RepetitionTime'], items['InversionTime']),
                ' The image is acquired using an echo time of {:.1f} ms, a repetition time of {:.1f} ms.'.format(items['EchoTime'], items['RepetitionTime']),
            ]
        else:
            content = [' The image is acquired using an echo time of {:.1f} ms, a repetition time of {:.1f} ms.'.format(items['EchoTime'], items['RepetitionTime'])]
        prompt.append({
            'content': content,
            'mask': 0, # 1 for necessary, 0 for random, -1 for omit
        })
    
    return prompt


def prompt_preprocess(info):
    prompt = ' The image preprocessing consists of '
    plist = []
    
    if 'Preprocess' in info:
        plist.append(', '.join(info['Preprocess']))

    if len(plist)==0:
        return []
    else:
        return [{'content': [prompt + ', '.join(plist) + '.'], 'mask': 1}]